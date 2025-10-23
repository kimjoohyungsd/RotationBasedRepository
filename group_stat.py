import copy
import logging
import math
import pprint
import time 
import tqdm
import functools
import os
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis

import torch
import torch.nn as nn
import tqdm
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizerFast, pipeline

from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils, model_utils, smooth_quant
from utils.process_args import process_args_ptq
from eval_utils import gptq_utils, rotation_utils, modeling_llama
from train_utils import apply_r3_r4


# Fake Quantization을 진행하는 함수 (일단 Symmetric한 경우만 가정을 하자)
def fake_quant(tensor,bits=8,eps=1e-12): # Tensor shape [Token NUm, hidden_dimension]
    x=np.asarray(tensor,dtype=np.float32)
    # print(x.shape)
    xmax=np.max(x,axis=1) # [T]
    xmin=np.min(x,axis=1) # [T]
    maxq= (1 << (bits-1)) - 1  

    amax = np.max(np.abs(x),axis=1)
    amax = np.maximum(amax,eps)
    scale = amax / maxq

    q=np.round(x/scale[:,None])
    q=np.clip(q,-(maxq+1),maxq)

    return q*scale[:,None]

def plot_group_boxplot(tensor, group_Num, g, name,layer_idx,file_path):
    # tensor shape: [Token_Num, Group_Num, Group_size]
    # torch → numpy 변환
    data = tensor.detach().cpu().numpy()  # [T, G, g]
    data = np.transpose(data,(1,0,2)) # [G,T,g]
    data = np.reshape(data, (np.size(data,0),-1)) # [G,T*g]
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
    plt.ylim(-4,4)

    plt.title(f"Layer: {layer_idx}, Module: {name},  Group size: {g}")
    plt.xlabel("Group Index")
    plt.ylabel("Activation Value")
    
    plt.legend(loc='upper right')
    plt.savefig(f'{file_path}.png')
    plt.show()
    plt.close()

def plot_group_kurtosis(tensor, group_Num, g, name,layer_idx,file_path):
    data = tensor.detach().cpu().numpy()  # [T, G, g]
    
    data = np.transpose(data,(1,0,2)) # [G,T,g]
    kurt=kurtosis(data,axis=2) # [G,T]
    kurt_mean=np.mean(kurt,axis=1) # [G]
    x=np.arange(kurt_mean.shape[0])
    plt.plot(x,kurt_mean)

    plt.title(f"Layer: {layer_idx}, Module: {name},  Group size: {g}")
    plt.xlabel("Group Index")
    plt.ylabel("Kurtosis Value")
    
    plt.legend(loc='upper right')
    plt.savefig(f'{file_path}_kurt.png')
    plt.show()
    plt.close()

def plot_in_group_dist(tensor, group_wise, g, name,layer_idx,file_path):
    data = tensor.detach().cpu().numpy()  # Group_wise: [T, G, g], Channel_wise: [T,g]
    
    if group_wise:
        group_idx = 6
        data = np.transpose(data,(1,0,2)) # [G,T,g]

        data = data[group_idx] # [T,g]
    # print(data.shape)
    # data = np.reshape(data,(np.size(data)))

    # Step1: Kurtosis를 구한다
    kurt=kurtosis(data,axis=1) # [T]
    kurt=np.mean(kurt).item() #

    # Step2: Quantization Error를 구한다
    re_quant8=fake_quant(data,bits=8) # [T,g]
    re_quant4=fake_quant(data,bits=4) #
    dif8 = LA.norm(data - re_quant8,ord=2,axis=1) # [T]
    dif4 = LA.norm(data - re_quant4,ord=2,axis=1) # [T]
    err8= np.mean(dif8).item()
    err4= np.mean(dif4).item()

    mins = np.min(data, axis=0)
    # print(mins)
    maxs = np.max(data, axis=0)
    # print(max)
    p1 = np.percentile(data, 1, axis=0)
    p99 = np.percentile(data, 99, axis=0)
    # print(p99)
    p25 = np.percentile(data, 25, axis=0)
    # print(p25)
    p75 = np.percentile(data, 75, axis=0)
    # print(p75)

    x = np.arange(data.shape[-1])

    plt.fill_between(x, mins, maxs, color="blue", alpha=0.2, label="Min/Max")
    plt.fill_between(x, p1, p99, color="red", alpha=0.4, label="1/99 Percentile")
    plt.fill_between(x, p25, p75, color="orange", alpha=0.6, label="25/75 Percentile")
    plt.ylim(-4,4)

    if group_wise:
        plt.title(f"Layer: {layer_idx}, Module: {name},  Group size: {g}")
    else: 
        plt.title(f"Layer: {layer_idx}, Module: {name},  Per-channel")

    plt.xlabel("Element_index")
    plt.ylabel("Activation Value")
    
    stats_text = (
        f"Avg. Quant Error (4-bit): {err4:.4f}\n"
        f"Avg. Quant Error (8-bit): {err8:.4f}"
    )
    plt.text(
        0.98, 0.98, 
        stats_text, 
        transform=plt.gca().transAxes, # Axes 좌표계를 기준으로 위치 지정 (0에서 1)
        fontsize=10, 
        verticalalignment='top', 
        horizontalalignment='right',
        bbox={'boxstyle': "round,pad=0.5", 'facecolor': 'white', 'alpha': 0.7} # 배경 상자 추가
    )

    plt.legend(loc='lower left')
    if group_wise:
        plt.savefig(f'{file_path}_group:{group_idx}.png')
    else:
        plt.savefig(f'{file_path}_per_channel.png')
    plt.show()
    plt.close()
    # data = np.reshape(data, (np.size(data,0),-1)) # [G,T*g]



    # path=os.path.join(draw_path,layer_idx)
# Step1: Rotate, Clip, Partition 논문에 기술된 대로 Hadamard 행렬을 적용하는 것이 weight를 Group Quantization 진행 시에 Per-channel에 average Kurtosis를 높이는지
# Step2: 해당 부분이 Quantization Error를 높이는지
# def get_act_scales(model, dataloader, num_samples=128):
#     model.eval()
#     device = next(model.parameters()).device
#     act_scales = {}

#     def stat_tensor(name, tensor):
#         hidden_dim = tensor.shape[-1]
#         tensor = tensor.view(-1, hidden_dim).abs().detach()
#         comming_max = torch.max(tensor, dim=0)[0].float().cpu()
#         if name in act_scales:
#             act_scales[name] = torch.max(act_scales[name], comming_max)
#         else:
#             act_scales[name] = comming_max

#     def stat_input_hook(m, x, y, name):
#         if isinstance(x, tuple):
#             x = x[0]
#         stat_tensor(name, x)

#     hooks = []
#     for name, m in model.named_modules():
#         if isinstance(m, nn.Linear):
#             hooks.append(
#                 m.register_forward_hook(
#                     functools.partial(stat_input_hook, name=name)))

#     for i in tqdm(range(num_samples)):
#         model(dataloader[i][0].to(device))

#     for h in hooks:
#         h.remove()

#     return act_scales

def main() -> None:
    model_args, training_args, ptq_args = process_args_ptq()

    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )

    # Step1: 모델 전처리하는 과정
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

    if ptq_args.smooth_quant:
        ptq_args.act_scales = f"./act_scales/{model_args.input_model.split('/')[-1]}.pt"
        print("Smoothing Applied")
        print(f"smoothing Alpha: {ptq_args.alpha}")
        if ptq_args.attention:
            print("Smoothing Applied to Attention")
        act_scales = torch.load(ptq_args.act_scales)
        smooth_quant.smoothing(model,ptq_args,act_scales)

    if (ptq_args.rotate):
        print("Rotation Applied")
        fuse_norm_utils.fuse_layer_norms(model) #
        rotation_utils.rotate_model(model, ptq_args)
        if not ptq_args.offline:
            apply_r3_r4.rotate_model(model, ptq_args)
            utils.cleanup_memory(verbos=True)
            
    
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda()


    model.seqlen = training_args.model_max_length
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
    inputs =tokenizer(full_text,return_tensors='pt',truncation=True, max_length=1000)

    model.eval()
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens=model.model.embed_tokens.cuda()
    layers[0] = layers[0].cuda()

    input_ids = inputs.input_ids.cuda()  # (1, text_len)
    dtype = next(iter(model.parameters())).dtype #
    # inps = torch.zeros(input_ids.shape,dtype=dtype).cuda()
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

    

    if ptq_args.stats_path is None:
        ptq_args.stats_path ='/home/jhkcool97/SpinQuant/stats'
    model_net=model_args.input_model.split('/')[-1]
    path = os.path.join(ptq_args.stats_path,model_net)
    os.makedirs(path,exist_ok=True)

    subpath=str()
    if ptq_args.smooth_quant:
        subpath+='smooth_quant,'
    if ptq_args.rotate:
        subpath+='rotate,' 
        # path = os.path.join(path,'rotate')
        if ptq_args.diagonal:
            subpath+=f'diagonal, diagonal_size:{ptq_args.diagonal_size}'
            # path = os.path.join(path,f'diagonal {ptq_args.diagonal_size}')

    group_size=str()
    if ptq_args.a_groupsize == -1:
        group_size = 'row'
    else:
        group_size = f'group_size:{ptq_args.a_groupsize}'

    subpath+=group_size
    path = os.path.join(path,subpath)
    os.makedirs(path,exist_ok=True)
    draw_path = os.path.join(path,'draw')

    # Rotation 적용 유무를 구현
    layer_list=[0,10,20,30]
    group_size=[32,64,128,256]
    Input_prefix="model.layers"
    act_stats={}

    def stat_input_hook(m, x, y, name,index):
        if isinstance(x, tuple):
            x = x[0]
        # stat_tensor(name, x)
        # for g in group_size:
        if ptq_args.a_groupsize == -1:
            g = x.shape[-1]
        else:
            g=ptq_args.a_groupsize
            
        tensor=x[0].detach().cpu() # [Token_Num, Dim_size]
        group_Num = tensor.shape[1] // g
        tensor=tensor.view(tensor.shape[0], -1, g) # [Token_Num, Group_Num, Group_size]
        tensor=tensor.squeeze()
        print(tensor.shape) 
        # 모델의 Stat을 구하는 부분 하지만 당장 중요한 것이 아니라 일단 주석처리를 하자
        
          
        # group_avg = torch.mean(torch.mean(tensor,dim=2),dim=0) # [Token_Num, Group_Num] => [group_Num]
        # group_var = torch.var(tensor, dim=(0,2),unbiased=False) # [Group_Num]

        #     # ===== Kurtosis per group (NumPy 버전; Torch 버전도 아래 제공) =====
            
        # xn =tensor.numpy().astype(np.float64)                # [T, group_Num, g]
        # mu = xn.mean(axis=(0, 2), keepdims=True)          # [1, group_Num, 1]
        # xc = xn - mu
        # m2 = (xc ** 2).mean(axis=(0, 2))                  # [group_Num]
        # m4 = (xc ** 4).mean(axis=(0, 2))                  # [group_Num]
        # eps = 1e-12
        # kurt = (m4 / (m2**2 + eps)) - 3.0                 # Fisher excess -> [group_Num]
        # kurt=kurt.mean().item()
        
        # ch_avg=0
        # channel_avg=torch.median(tensor,dim=0).values # [Group_Num, Group_size]
        # # channel_avg=torch.mean(tensor,dim=0) # [Group_Num, Group_size]
        # print("Kurt:"+str(kurt))
        #     # 임계값: 그룹 평균 + 3*표준편차
        # thr = group_avg + 3.0 * torch.sqrt(group_var + 1e-12)   # [group_Num]

        # # 브로드캐스트해 비교 -> [group_Num, g]
        # over = (channel_avg > thr.unsqueeze(-1)).float()
        # ch_avg = (over.mean(dim=1)).mean().item()  # 또는 그룹별로 보고 싶으면 mean(dim=1)만 사용
        # print('ch_avg: '+str(ch_avg))
        # key = f"{Input_prefix}.{index}.{name}, group_size:{g}"
        # act_stats[key] = {
        # "outlier_avg": ch_avg,
        # "kurtosis": kurt,          # [group_Num] 리스트로 저장
        # # "group_avg": group_avg.tolist(),    # 참고용
        # # "group_var": group_var.tolist(),    # 참고용
        # }

        if index in layer_list:
            file_path = os.path.join(draw_path,str(index))
            os.makedirs(file_path,exist_ok=True)
            file_path += f'/{index}.{name}.{g}'
            # plot_group_boxplot(tensor,group_Num,g,name,index,file_path)
            # plot_group_kurtosis(tensor,group_Num,g,name,index,file_path)
            if ptq_args.a_groupsize == -1:
                group_wise=False
            else:
                group_wise=True
            plot_in_group_dist(tensor,group_wise,g,name,index,file_path)
            # tensor=torch.permute(tensor,(1,0,2)) # [Token,Group_Num, Group_size] => [Group_NUm, Tok]
            # tensor=tenosr.view(tensor.shape[0],-1) # [Group_Num, Token*Group_size]
            # blocks = []
            # values = []
            # block_prefix='blocks: '
            # for block in range(group_Num):
            #     layers 
            # 
            # 특정 Group  

    input_list=['self_attn.q_proj','self_attn.o_proj','mlp.up_proj','mlp.down_proj']

    for i in tqdm.tqdm(range(len(layers)),desc="Capturing Information Block wise"):
        # Step 3-1: 각 Layer의 channel 단위의 Weight Kurtosis를 구한다
        layer = layers[i].cuda()
        hooks=[]
        for name, m in layer.named_modules():
            # print("registering hook for",name, type(m))
            if isinstance(m,nn.Linear) and name in input_list:
                # hooks를 추가한다
                # print("registering hook")
                hooks.append(
                    m.register_forward_hook(functools.partial(stat_input_hook,name=name,index=i))
                )
        outs=layer(inps,attention_mask=attention_mask,position_ids=position_ids)[0]
        print(outs.shape)
        for h in hooks:
            h.remove()
        inps=outs
        layer.cpu()
    
    



if __name__ == '__main__':
    main()

