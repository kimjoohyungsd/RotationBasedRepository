import torch
import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoConfig
)
import transformers

import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm

from utils import data_utils
from utils.process_args import process_args_ptq

def get_input_device(model):
    # With --distribute the model is sharded over several GPUs; token ids must be fed to the
    # device that holds embed_tokens (accelerate hooks move activations onward from there).
    return model.get_input_embeddings().weight.device

def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = get_input_device(model)
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules(): # model.layers.0.self_attn.q_proj
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook( # functools_partial 고유의 훅 함수 만듬
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = get_input_device(model)
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))


    for h in hooks:
        h.remove()

    return act_shifts

@torch.no_grad()
def main():
    model_args, training_args, ptq_args = process_args_ptq()
    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    # Same placement policy as ptq.py: --distribute shards the model over every visible GPU
    # (device_map="auto"), otherwise the whole model goes to a single GPU.
    device_map = "auto" if ptq_args.distribute else None
    max_memory = None
    if ptq_args.distribute:
        # cap each GPU so activations of the 2048-token calibration batch still fit
        max_memory = {i: "14GiB" for i in range(torch.cuda.device_count())}

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
        device_map=device_map,
        max_memory=max_memory,
    )
    if not ptq_args.distribute:
        model.cuda()

    # Llama-3 ships a tiktoken-style tokenizer.json, so it needs the generic fast tokenizer (as in ptq.py)
    tokenizer_cls = PreTrainedTokenizerFast if 'Llama-3' in model_args.input_model else LlamaTokenizerFast
    tokenizer = tokenizer_cls.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    ptq_args.net = model_args.input_model.split('/')[-1] # Llama-2-7b-hf
    dataloader=data_utils.get_wikitext2(nsamples=ptq_args.nsamples,tokenizer=tokenizer,eval_mode=False) # dataloader로 값을 읽어온다 [(tokenized.input_ids (shape(1,2048)),(tokenized.answer_labels))]
    act_scales = get_act_scales(model, dataloader,ptq_args.nsamples) # act_scales로 값을 읽어온다
    act_shifts = get_act_shifts(model,dataloader,ptq_args.nsamples) # act_shfits 값을 읽어온다

    # 읽어들인 값을 저장하는 과정
    scales_path = os.path.join(ptq_args.scales_output_path,f'{ptq_args.net}.pt')
    os.makedirs(os.path.dirname(scales_path), exist_ok=True)
    torch.save(act_scales, scales_path)

    shifts_path = os.path.join(ptq_args.shifts_output_path,f'{ptq_args.net}.pt')
    os.makedirs(os.path.dirname(shifts_path),exist_ok=True)
    torch.save(act_shifts, shifts_path)

if __name__ == '__main__':
    main()
    