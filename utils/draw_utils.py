import copy
import logging
import math
import pprint
import time 
import tqdm
import functools
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from tqdm import tqdm
from functools import partial

from utils import quant_utils


def plot_3d_map(tensor,file_path,layer_idx,name,is_weight=False):
    data = tensor.detach().abs().cpu().float().numpy()

    if is_weight is False:
        data = data.reshape(-1,data.shape[-1]) # Activation: [Batch*Token,channel]
        T, C = data.shape
        
        x_label = "Channel"
        y_label = "Token"

    else:
        T, C = data.shape

        x_label = "Input Channel"
        y_label = "Output Channel"

    # data = data.T

    x = np.arange(C)
    y = np.arange(T)

    X,Y = np.meshgrid(x,y)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X,
        Y,
        data,
        cmap="coolwarm"
    )

    ax.set_title(f"layers.{layer_idx}.{name}", fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_zlabel("|Value|")

    # ax.view_init(elev=elev, azim=azim)

    fig.colorbar(surf, shrink=0.5, aspect=12, pad=0.1)
    plt.tight_layout()

    plt.savefig(f"{file_path}/3d_plot.png", dpi=300)
    plt.close()
    
def plot_boxplot(tensor,file_path,layer_idx,name,is_weight=False):
    # tensor shape: [Token_Num, Group_Num, Group_size]
    # torch → numpy 변환
    data = tensor.detach().cpu().numpy()  # Weight: [out_c,in_c], Activation: [Batch, Seq, ch]
    if is_weight:
        data = data.reshape(data.shape[0],-1)
    else:
        B, S, D = data.shape
        data = data.reshape(-1, D)

    data = data.T
    # print(np.isnan(data))
    mins = np.min(data, axis=1)
    # print(mins)
    maxs = np.max(data, axis=1)
    # print(max)
    p1 = np.percentile(data, 1, axis=1)
    p99 = np.percentile(data, 99, axis=1)
    # print(p99)
    p25 = np.percentile(data, 25, axis=1)
    # print(p25)
    p75 = np.percentile(data, 75, axis=1)
    # print(p75)

    x = np.arange(data.shape[0])

    plt.fill_between(x, mins, maxs, color="blue", alpha=0.2, label="Min/Max")
    plt.fill_between(x, p1, p99, color="red", alpha=0.4, label="1/99 Percentile")
    plt.fill_between(x, p25, p75, color="orange", alpha=0.6, label="25/75 Percentile")
    # plt.ylim(-4,4)

    plt.title(f"Layer: {layer_idx}, Module: {name}")
    plt.xlabel("Group Index")
    plt.ylabel("Channel Value")
    
    plt.legend(loc='upper right')
    plt.savefig(f'{file_path}/box_plot.png')
    plt.show()
    plt.close()


def draw_weight(model,save_path,args):
    target_layers = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'
    ]
    for i in tqdm(range(len(model.model.layers)), desc="Visualizing Layers"):
        layer = model.model.layers[i]

        qlinear_modules = quant_utils.find_qlayers(layer)
        layer_path = os.path.join(save_path,f"layers: {i}")
        os.makedirs(layer_path,exist_ok=True)

        for name, module in qlinear_modules.items():
            
            if name not in target_layers:
                continue
            
            if isinstance(module,quant_utils.ActQuantWrapper):
                module = module.module

            tmp_path = os.path.join(layer_path,name)
            os.makedirs(tmp_path,exist_ok=True)

            #1. QuaRot 형식의 BoxPlot 그래프를 그린다
            plot_boxplot(module.weight,tmp_path, i, name,True)
            print(f"Drawing Boxplot for {name} is finished")
            #2. 3차원 형식의 활성화 그림을 그린다
            plot_3d_map(module.weight,tmp_path,i,name,True)
            print(f"Drawing 3D plot for {name} is finished")

@torch.no_grad()
def draw_activations(model,save_path,args,testloader):

    # Step1: testloader에서 seq_len 2048의 길이에 token_ids를 추출
    i = random.randint(0, testloader.input_ids.shape[1] - model.seqlen - 1)
    j = i + model.seqlen
    inp = testloader.input_ids[:,i:j].to(device=model.device)
    layers = model.model.layers
    # Step2: testloader에서 hook 함수 생성
    def stat_input_hook(m, x, y, layer_save_path, name, index):
        if isinstance(x, tuple):
            x = x[0]

            plot_boxplot(x,layer_save_path, index,name,False)

            plot_3d_map(x,layer_save_path,index,name,False)
    
    # Step3: Catcher라는 리스트 생성
    inps = [None] * len(layers) 
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "position_embeddings": None}
    class Catcher(torch.nn.Module): # catcher
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError
    
    layers[0] = Catcher(layers[0])

    if hasattr(layers[0].module, "attention_type"):
        layers[0].attention_type = layers[0].module.attention_type

    try:
        model(inp.to(device=model.device))
    except ValueError:
        pass

    layers[0] = layers[0].module
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    position_embeddings = cache['position_embeddings']

    torch.cuda.empty_cache()

    # Step4: Activation에 그림을 포착하는 단계
    target_layers = [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
        'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'
    ]
    cur_inp = inps[0]
    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i]
        layer_device = next(layer.parameters()).device
        layer_path = os.path.join(save_path,f"layers: {i}")
        os.makedirs(layer_path,exist_ok=True)
        hooks=[]
        for name, module in layer.named_modules():

            if name not in target_layers:
                continue

            if isinstance(module,quant_utils.ActQuantWrapper):
                module = module.module
            
            tmp_path = os.path.join(layer_path,name)
            os.makedirs(tmp_path,exist_ok=True)
            hooks.append(module.register_forward_hook(functools.partial(stat_input_hook, layer_save_path=tmp_path, name=name, index=i)))
        
        out=layer(cur_inp.to(device=layer_device),attention_mask=attention_mask,position_ids=position_ids,position_embeddings=position_embeddings)

        for h in hooks:
            h.remove()
        
        if isinstance(out, tuple):
            out = out[0]
        
        cur_inp = out



