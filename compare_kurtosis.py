import copy
import logging
import math
import pprint
import time 
import tqdm

import torch
import torch.nn as nn
import tqdm
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast, pipeline

from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils, model_utils
from utils.process_args import process_args_ptq
from eval_utils import gptq_utils, rotation_utils
from train_utils import apply_r3_r4

# Step1: Rotate, Clip, Partition 논문에 기술된 대로 Hadamard 행렬을 적용하는 것이 weight를 Group Quantization 진행 시에 Per-channel에 average Kurtosis를 높이는지
# Step2: 해당 부분이 Quantization Error를 높이는지

def main() -> None:
    model_args, training_args, ptq_args = process_args_ptq()

    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )

    # Rotation 적용 유무를 구현
    group_size=[32,64,128,256]
    bits_list=[2,4,8]
    if (ptq_args.rotate):
        print("Rotation Applied")
        fuse_norm_utils.fuse_layer_norms(model) #
        rotation_utils.rotate_model(model, ptq_args)
        if not ptq_args.offline:
            apply_r3_r4.rotate_model(model, ptq_args)
            
            utils.cleanup_memory(verbos=True)
            quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
            qlayers = quant_utils.find_qlayers(model) # Quantized 된 layer의 dictionary format을 만든
            for name in qlayers:
                if "down_proj" in name and not ptq_args.offline:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = ptq_args.fp32_had
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": ptq_args.k_bits,
                "k_groupsize": ptq_args.k_groupsize,
                "k_sym": not (ptq_args.k_asym),
                "k_clip_ratio": ptq_args.k_clip_ratio,
            }
            
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )
    
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    tokenizer = LlamaTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_args.input_model,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
    add_eos_token=False,
    add_bos_token=False,
    token=model_args.access_token,
    )


    # Step2: Dataset을 바탕으로 하여 각 Weight Tensor의 정보 준비를 완성한다
    texts = data_utils.load_example_dataset() # 가지고 올때 [text sample1, text sample2, text sample3, text sample4, text sample5] 형식의 list 형태를 가지고 온다
    full_text = " ".join(texts)
    inputs =tokenizer(full_text,return_tensors='pt',truncation=True, max_length=1500)

    model.eval()
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens=model.model.embed_tokens.cuda()
    layers[0] = layers[0].cuda()

    input_ids = inputs.input_ids.cuda()  # (1, text_len)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(input_ids.shape,dtype=dtype).cuda()
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module): # catcher
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.captured_input = None

            def forward(self, inp, **kwargs):
                
                # cache["i"] += 1
                self.captured_input = inp
                cache["attention_mask"] = kwargs["attention_mask"]
                cache["position_ids"] = kwargs["position_ids"]
                raise ValueError

    layers[0] = Catcher(layers[0])

    try:
        model(input_ids)
    except ValueError:
        pass

    inps = layers[0].captured_input
    print(inps.shape)
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    torch.cuda.empty_cache()
    model.model.embed_tokens = model.model.embed_tokens.cpu()

    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]

    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    # Step 3: 각 weight Tensor 단위로 1) Weight의 Kurtosis 2) Output의 Quantization Error를 분석한다 
    for i in tqdm.tqdm(range(len(layers)),desc="Capturing Information"):
        # Step 3-1: 각 Layer의 channel 단위의 Weight Kurtosis를 구한다
        