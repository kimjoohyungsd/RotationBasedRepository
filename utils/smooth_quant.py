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

def smooth_v_o(v_proj,o_proj,scales,head_dim):
    """Smooth the v_proj -> o_proj pair, with or without grouped-query attention.

    scales has o_proj.in_features entries (one per attention-head channel), but v_proj only
    has num_kv_heads * head_dim output channels: each kv head is shared by n_rep query heads
    (repeat_kv), so o_proj input channel j = (g * n_rep + r) * head_dim + d reads v_proj output
    channel g * head_dim + d. For GQA (e.g. Llama-3.1-8B: v_proj [1024,4096], o_proj [4096,4096])
    the n_rep scales that share one v_proj channel are arithmetic-averaged into a single scale
    s_v. v_proj rows are divided by s_v and o_proj columns are multiplied by s_v repeated over
    the n_rep query heads (block-diagonal along the heads), which keeps the layer pair
    function-preserving. Multiplying o_proj by the un-averaged scales would not.
    """
    if v_proj.out_features == o_proj.in_features: # MHA: one scale per channel
        smooth_fc_fc(v_proj,o_proj,scales)
        return

    n_kv = v_proj.out_features // head_dim # Number of KV head
    n_rep = o_proj.in_features // v_proj.out_features # Number of KV repetition
    assert v_proj.out_features % head_dim == 0 and o_proj.in_features == n_kv * n_rep * head_dim, \
        "v_proj {} / o_proj {} do not match head_dim {}".format(tuple(v_proj.weight.shape),tuple(o_proj.weight.shape),head_dim)

    scales = scales.float().view(n_kv,n_rep,head_dim) # [Number of KV head, Repetition, Head Dim]
    scales_v = scales.mean(dim=1)                                          # [n_kv, head_dim], averaged over the n_rep query heads
    scales_o = scales_v.unsqueeze(1).expand(n_kv,n_rep,head_dim).reshape(-1)  # [n_kv*n_rep*head_dim], repeated back per query head
    smooth_fc_fc_gqa(v_proj,o_proj,scales_v.reshape(-1),scales_o)

def smooth_fc_fc_gqa(fc1,fc2,scales_1,scales_2):
    with torch.no_grad():
        fc1.weight.div_(scales_1.to(device=fc1.weight.device,dtype=fc1.weight.dtype).view(-1,1))
        fc2.weight.mul_(scales_2.to(device=fc2.weight.device,dtype=fc2.weight.dtype).view(1,-1))

def smoothing(model,args,act_scales):
    pairs = { # "레이어 그룹핑 가이드"
            "q_proj":"qkv", # q_proj의 통계치를 보고 'qkv'라는 이름의 스케일을 만든다
            "o_proj":"out", # o_proj의 통계치를 보고 'out'이라는 이름의 스케일을 만든다
            "up_proj":"fc1",# up_proj의 통계치를 보고 'fc1'이라는 이름의 스케일을 만든다
            "down_proj":"down", # down_proj의 통계치를 보고 'down'이라는 이름의 스케일을 만든다
        }
    
    
    layer_name_prefix = "model.layers"
    CLIPMIN = 1e-5 # 1e-5
    CLIPMAX = 1e4 # 1e4

    
    layers=model.model.layers
    for i in range(len(layers)): # 1. 각 DecoderLayer 단위로 forward pass 진행
        layer=layers[i]
        dev = next(layer.parameters()).device
        scales={} # 계산된 보정값 저장소
        for name, module in layer.named_modules(): # 2. 각 Decoderlayer에 특정 Linear Layer의 Activation scales 값 추출
            if isinstance(module, nn.Linear):
                for key in pairs.keys(): # 3. 해당하는 Activation scales의 해당하는 값 추출
                    if key in name:
                        # print(name)
                        dtype=module.weight.dtype
                        act=act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev,dtype=dtype).clamp(min=CLIPMIN) # 3-1: activation에 통계값 load
                        weight = module.weight.abs().max(dim=0)[0].clamp(min=CLIPMIN) # 3-2: Weight에 해당 Channel의 통계값 load
                        scale = (act.pow(args.alpha)/weight.to(act.device).pow(1-args.alpha)).clamp(min=CLIPMIN) # 3-3: SmoothQuant의 수식을 바탕으로 Scaling 값 구함
                        scales[pairs[key]]=scale # 3-4: Scales라는 Dictionary에 해당 값 구함 

        smooth_ln_fcs(layer.input_layernorm,[layer.self_attn.q_proj,layer.self_attn.k_proj,layer.self_attn.v_proj],scales["qkv"]) # Input Layernorm, [q_proj,k_proj,v_proj]에 적용
        smooth_ln_fcs(layer.post_attention_layernorm,[layer.mlp.gate_proj,layer.mlp.up_proj],scales["fc1"]) # post_attention_layernorm, mlp.gate_proj,mlp.up_proj의 적용
        smooth_fc_fc(layer.mlp.up_proj,layer.mlp.down_proj,scales["down"])
        if (args.attention):
            head_dim = getattr(model.config,"head_dim",None) or model.config.hidden_size // model.config.num_attention_heads
            smooth_v_o(layer.self_attn.v_proj,layer.self_attn.o_proj,scales["out"],head_dim)       
