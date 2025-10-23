import torch
import os

from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizerFast,
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

def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
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
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook( # func_tools_partial 고유의 훅 함수 만듬
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales

@torch.no_grad()
def main():
    model_args, training_args, ptq_args = process_args_ptq()
    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    model.cuda()
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
    ptq_args.net = model_args.input_model.split('/')[-1]
    dataloader=data_utils.get_wikitext2(tokenizer=tokenizer,eval_mode=False)
    act_scales = get_act_scales(model, dataloader,ptq_args.nsamples)
    save_path = os.path.join(ptq_args.scales_output_path,f'{ptq_args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

if __name__ == '__main__':
    main()
    