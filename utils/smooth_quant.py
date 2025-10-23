# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on Smoothquant (link= https://arxiv.org/pdf/2211.10438)

import torch.nn as nn
import torch

def smooth_ln_fcs(ln,fcs,scales):
    if not isinstance(fcs,list):
        fcs=[fcs]
    # if hasattr(ln, 'bias') and ln.bias is not None:
    #     ln.bias.div_(scales)
    with torch.no_grad():
        ln.weight.div_(scales)

    for fc in fcs:
        # if hasattr(fc, 'bias') and fc.bias is not None:
        #     fc.bias.add_(fc.weight@shifts.to(fc.weight.device))
        with torch.no_grad():
            fc.weight.mul_(scales.to(fc.weight.device).view(1,-1))

def smooth_fc_fc(fc1,fc2,scales):
    with torch.no_grad():

        fc1.weight.div_(scales.to(fc1.weight.device).view(-1,1))
        fc2.weight.mul_(scales.to(fc2.weight.device).view(1,-1))
def smoothing(model,args,act_scales):
    pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1",
            "down_proj":"down",
        }
    
    
    layer_name_prefix = "model.layers"
    CLIPMIN = 1e-5 # 1e-5
    CLIPMAX = 1e4 # 1e4

    dev=model.device
    layers=model.model.layers
    for i in range(len(layers)):
        layer=layers[i]
        scales={}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                for key in pairs.keys():
                    if key in name:
                        # print(name)
                        dtype=module.weight.dtype
                        act=act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev,dtype=dtype).clamp(min=CLIPMIN)
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=CLIPMIN)
                        scale = (act.pow(args.alpha)/weight.to(act.device).pow(1-args.alpha)).clamp(min=CLIPMIN)
                        scales[pairs[key]]=scale

        smooth_ln_fcs(layer.input_layernorm,[layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj],scales["qkv"])
        smooth_ln_fcs(layer.post_attention_layernorm,[layer.mlp.gate_proj,layer.mlp.up_proj],scales["fc1"])
        smooth_fc_fc(layer.mlp.up_proj,layer.mlp.down_proj,scales["down"])
        if (args.attention):
            smooth_fc_fc(layer.self_attn.v_proj,layer.self_attn.o_proj,scales["out"])       
