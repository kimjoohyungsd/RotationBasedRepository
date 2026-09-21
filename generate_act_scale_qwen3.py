"""SmoothQuant activation-scale generator for Qwen3 (generate_act_scale.py is Llama-only).

Saves per-input-channel abs-max of every nn.Linear input (WikiText-2 train, 128 x 2048 tokens)
to  act_scales/<net>.pt , which eval_utils.main.ptq_model loads when --smooth_quant is set.

    python generate_act_scale_qwen3.py --input_model Qwen/Qwen3-8B
"""
import argparse
import functools
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from eval_utils.modeling_qwen3 import Qwen3ForCausalLM
from utils import data_utils


@torch.no_grad()
def get_act_scales(model, dataloader, num_samples, input_device):
    model.eval()
    act_scales = {}

    def hook(m, x, y, name):
        x = x[0] if isinstance(x, tuple) else x
        cur = x.view(-1, x.shape[-1]).abs().detach().amax(dim=0).float().cpu()
        act_scales[name] = torch.max(act_scales[name], cur) if name in act_scales else cur

    hooks = [m.register_forward_hook(functools.partial(hook, name=n))
             for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(input_device))
    for h in hooks:
        h.remove()
    return act_scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_model", required=True)
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--output_dir", default="./act_scales/")
    a = ap.parse_args()

    model = Qwen3ForCausalLM.from_pretrained(
        a.input_model, torch_dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "17GiB", 1: "18GiB"})
    tokenizer = AutoTokenizer.from_pretrained(a.input_model, use_fast=True, add_eos_token=False, add_bos_token=False)
    loader = data_utils.get_wikitext2(nsamples=a.nsamples, seed=0, seqlen=2048,
                                      tokenizer=tokenizer, eval_mode=False)
    scales = get_act_scales(model, loader, a.nsamples, model.model.embed_tokens.weight.device)
    net = a.input_model.split("/")[-1]
    os.makedirs(a.output_dir, exist_ok=True)
    torch.save(scales, os.path.join(a.output_dir, f"{net}.pt"))
    print("saved", os.path.join(a.output_dir, f"{net}.pt"), len(scales), "linears")


if __name__ == "__main__":
    main()
