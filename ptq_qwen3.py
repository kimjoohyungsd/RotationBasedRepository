import datetime
from logging import Logger

import torch
# import torch.distributed as dist
from transformers import LlamaTokenizerFast,PreTrainedTokenizerFast,AutoTokenizer # LlamaForCausalLM, pipeline
import transformers
import os

from eval_utils.main import ptq_model
# from eval_utils.modeling_llama import LlamaForCausalLM
# from eval_utils.modeling_qwen2 import Qwen2ForCausalLM # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?

from eval_utils.modeling_qwen3 import Qwen3ForCausalLM
from utils import data_utils, eval_utils, utils, draw_utils
from utils.process_args import process_args_ptq

from datasets import load_dataset

from utils.process_args import process_args_ptq

def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)

    wandb_run = utils.setup_wandb(model_args.input_model, ptq_args)
    if wandb_run is not None:
        log.info("wandb run: {} ({})".format(wandb_run.name, wandb_run.url))
    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 or config.torch_dtype==torch.bfloat16 else torch.float16
    if ptq_args.draw:
        dtype = torch.float16

    device_map = "auto" if ptq_args.distribute else None
    n_gpus = torch.cuda.device_count()

    # 0번 GPU에는 파라미터를 적게(예: 16GB), 나머지는 넉넉히(예: 22GB) 할당
    # 70B 모델은 전체 약 140GB(FP16) / 35GB(W4)이므로 이에 맞춰 분배
    max_memory = {}  # 빈 딕셔너리로 초기화
    if ptq_args.distribute:
        n_gpus = torch.cuda.device_count()
        # 0번 GPU는 Activation 공간 확보를 위해 적게 할당
        max_memory[0] = "17GiB" 
        for i in range(1, n_gpus):
            # 나머지 GPU는 모델 파라미터를 담기 위해 더 넉넉히 할당 (예: 24GB 카드 기준)
            max_memory[i] = "18GiB" 
    else:
        max_memory = None # 분산 모드가 아닐 때는 None 전달

    model_args.net= model_args.input_model.split('/')[-1]
    if 'Qwen3' in model_args.net:
        model = Qwen3ForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
            device_map=device_map
        )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    if not ptq_args.distribute:
        model.cuda() # 모델을 GPU로 옮긴다

    if (ptq_args.rotate):
        log.info("Rotation applied")
        if ptq_args.optimized_rotation_path is not None:
            log.info("Trained Rotation Matrix applied")

    log.info("Quantization bits W: {},A: {}, KV: {}".format(ptq_args.w_bits,ptq_args.a_bits,ptq_args.k_bits))

    a_groupsize = ptq_args.a_groupsize if ptq_args.a_groupsize != -1 else "per-token"
    w_groupsize = ptq_args.w_groupsize if ptq_args.w_groupsize != -1 else "per-channel"
    kv_groupsize = ptq_args.k_groupsize if ptq_args.k_groupsize != -1 else 128
    log.info("Quantization group size W: {}, A: {}, KV: {}".format(a_groupsize,w_groupsize,kv_groupsize))
    
    if ptq_args.per_column:
        log.info("Quantization is done on column wise manner")

    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_args.input_model,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            add_eos_token=False,
            add_bos_token=False,
            token=model_args.access_token,
        )
    
    model = ptq_model(ptq_args, model, log, tokenizer,model_args) # 
    model.seqlen = training_args.model_max_length

    log.info("Complete tokenizer loading...")

    results = {}
    if ptq_args.wikitext2 or ptq_args.draw:
        model.config.use_cache = False
        testloader = data_utils.get_wikitext2( 
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        if ptq_args.wikitext2:
            dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
            results['wiki2_ppl'] = dataset_ppl

        if ptq_args.draw:
            if ptq_args.distribution_dir is None:
                ptq_args.distribution_dir = os.path.join("figures", model_args.net)

            os.makedirs(ptq_args.distribution_dir, exist_ok=True)

            save_path = ptq_args.distribution_dir
            if ptq_args.rotate:
                if ptq_args.optimized_rotation_path:
                    save_path = os.path.join(save_path,"Rotated (SpinQuant)")
                else:
                    save_path = os.path.join(save_path,"Rotated ({})".format(ptq_args.rotate_mode))
            elif ptq_args.smooth_quant:
                save_path = os.path.join(save_path,"Smoothed")

            if ptq_args.weight_check:

                weight_path = os.path.join(save_path,"Weight")
                os.makedirs(weight_path,exist_ok=True)

                # draw_utils에서 모델의 분포 및 3차원 그래프를 그리는 것을 진행하지
                draw_utils.draw_weight(model,weight_path,ptq_args)

            if ptq_args.act_check:
                act_path = os.path.join(save_path,"Act")
                draw_utils.draw_activations(model,act_path,ptq_args,testloader)


        # dist.barrier()


    if ptq_args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        if not ptq_args.distribute:
            model.cuda() # 모델을 GPU로 옮긴다
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=ptq_args.lm_eval_batch_size)

        task_names = ptq_args.tasks

        
        for task_name in task_names:
            log.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=ptq_args.lm_eval_batch_size)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            log.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        log.info(metric_vals)
        log.info(results)

if __name__ == "__main__":
    train()
