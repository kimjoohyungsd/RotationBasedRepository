import datetime
from logging import Logger
import json
import os
import re
import numpy as np
from collections import defaultdict
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import glob

import torch
from transformers import LlamaTokenizerFast, pipeline
import transformers

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils, model_utils
from utils.process_args import process_args_ptq
from train_utils import apply_r3_r4
from eval_utils import gptq_utils, rotation_utils

def average_list(x) -> float:
    return sum(x)/len(x)

def kurtosis_stats(x) -> tuple[float,float]:
    """행(또는 블록)별 커토시스 평균"""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("kurtosis_stats expects 2D array")
    row_std = x.std(axis=1)
    valid = row_std > 0
    x_use = x[valid] if np.any(valid) else x
    k = stats.kurtosis(x_use, axis=1, fisher=True, bias=False, nan_policy='omit')

    return float(np.mean(k)) if k.size else float("nan"), float(np.mean(k>=0))

def safe_group_view(x, g: int):
    """마지막 축을 g로 정확히 나누어떨어지도록 앞쪽만 사용해 [*, g] 블록으로 변환"""
    x = np.asarray(x, dtype=np.float64)
    usable = (x.shape[-1] // g) * g
    if usable == 0:
        return None
    return x[:, :usable].reshape(-1, g)

def compute_stats_for_matrix(arr: np.ndarray, bits_list, sym: bool, group_sizes):
    """
    arr: torch.Tensor or np.ndarray, shape [..., last_dim]
    반환: {
      'per-row': {'Kurtosis': float, 'qerr': {'2-bit': .., '4-bit': .., ...}},
      'group_size: 128': {...}, ...
    }
    """
    x0 = np.asarray(arr.reshape(-1, arr.shape[-1]).cpu().numpy() if hasattr(arr, "cpu") else arr, dtype=np.float64)

    out = {}
    # per-row
    mean_excess, prop_ge3 = kurtosis_stats(x0)
    out["per-row"] = {
        "Kurtosis": {"mean_excess": mean_excess,"prop_pearson_ge3":prop_ge3},
    }
    # group-wise
    for g in group_sizes:
        key = f"group_size {g}"
        xg = safe_group_view(x0, g)
        if xg is None:
            out[key] = {"Kurtosis": float("nan"),
                        ""
                        "qerr": {f"{b}-bit": float("nan") for b in bits_list}}
            continue
        mean_ex_g,p_ge3_g = kurtosis_stats(xg)
        out[key] = {
            "Kurtosis": {"mean_excess":mean_ex_g,"prop_pearson_ge3":p_ge3_g},
        }
    return out    

def aggregate_kurtosis(arr:np.ndarray) -> tuple[float,int]:
    arr =np.asarray(arr,dtype=np.float64)
    row_std = arr.std(axis=1)
    valid = row_std > 0
    x_use = arr[valid] if np.any(valid) else arr
    k = stats.kurtosis(x_use, axis=1, fisher=True, bias=False, nan_policy='omit')

    # k_filtered = k[~np.isnan(k)]

    return np.sum(k).item(), len(k)
    
def make_model_kurtosis_aggregator(group_sizes):
    agg = {
        "per-row": {"sum": 0.0, "cnt": 0},                    # 행 단위
        **{"group_size "+str(g): {"sum":0.0,"cnt":0} for g in group_sizes}  # g별 그룹 단위
    }
    return agg

# 해당 파일은 fake quantization을 적용하는 것이 아니라 실제 Rotation 적용 유무에 따라서 Weight 와 Activation의 Distribution을 보기 위한 함수로 별도로 진행하면된다
def train() -> None:

    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)

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

    transformers.set_seed(ptq_args.seed)
    model.eval()

    # ptq_args.rotate=False
    # Step 1: 모델을 Rotation을 적용하는 경우에는 위와 같이 미리 적용을 한다
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

    model.cuda() # 모델을 GPU로 옮긴다

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

    # Case 2: 실제 Weight와 Activation의 Distribution을 Profiling을 진행하기 위한 부분을 진행한다

    # Overview Step:
        # Step 1: Weight의 Distribution을 먼저 저장할지 부터 정한다
        # Step 2: Activation의 Hook 함수를 추가한다 
        # Step 3: 특정 Dataset에서의 함수를 추가한다
        # Step 4: 실제 Runtime 시에서 해당 값을 가지고 온다
    
    # 해당 Task를 진행하기 전에 경로 설정을 미리 진행한다
    # 1. 저장 경로(save_path)를 결정하고 디렉터리 생성


    if ptq_args.distribution_dir is None: 
        save_path = "Outlier"
    else:
        save_path = ptq_args.distribution_dir
        
    os.makedirs(save_path, exist_ok=True)

    
# Append a sub-directory based on rotation state
    if ptq_args.rotate: # root/rotatted (or Unrotated) /(Rotated_path)/
        sub_dir = "rotated"
        if ptq_args.optimized_rotation_path is not None:
            file_name = os.path.basename(ptq_args.optimized_rotation_path)
            sub_dir = os.path.join(sub_dir, file_name)
        else:
            sub_dir = os.path.join(sub_dir, "hadamard")
    else:
        sub_dir = "unrotated"

    # Combine the base path with the sub-directory
    save_path = os.path.join(save_path, sub_dir) # save_path= root / (Rotated_or Unrotated)
    os.makedirs(save_path, exist_ok=True)
    
    visualize_path=os.path.join(save_path,"Visualize")
    os.makedirs(visualize_path,exist_ok=True)

    dist_path = os.path.join(save_path,"Dist")
    os.makedirs(dist_path,exist_ok=True)


    visualize_index=[0,10,20,30]
    group_size=[32,64,128,256]
    bits_list=[2,4,8]


# Step 1: Weight의 값을 저장하는 것이 아니라, Weight의 통계값을 저장한다
    model.cpu()
    if ptq_args.weight_check:

        weights_tensor = defaultdict(lambda:defaultdict(lambda: defaultdict(dict))) # key 1: tensor name Key 2: Group Size Key 3: Kurtosis and qerr

        weights_layer = defaultdict(lambda:defaultdict(lambda: defaultdict(dict))) # key 1: layer_index key2: Group Size Key 3: Kurtosis
         # key 1: group_size value: List

        # weights_model = defaultdict(lambda:defaultdict()) # key1 Group size Key 2: Kurtosis
        # weights_model_list = defaultdict(list)

        weights_visualize_path=os.path.join(visualize_path,'Weights')

        os.makedirs(weights_visualize_path, exist_ok=True)


        

        # Step 1: Embed_tokens를 Weight단위에서 Profiling을 진행한다

        embed_tokens= model.model.embed_tokens
        weights_tensor['embed_tokens'] = compute_stats_for_matrix(embed_tokens.weight.data.detach().cpu().numpy(),bits_list,True,group_size)
        weights_layer['embed_tokens'] = compute_stats_for_matrix(embed_tokens.weight.data.detach().cpu().numpy(),bits_list,True,group_size)
        ## model에서 group size 단위로 Sum 하는 부분을 구현한다 argument 1: group_size, argument2: 



        lm_head = model.lm_head
        weights_tensor['lm_head'] = compute_stats_for_matrix(lm_head.weight.data.detach().cpu().numpy(),bits_list,True,group_size)
        weights_layer['lm_head'] = compute_stats_for_matrix(lm_head.weight.data.detach().cpu().numpy(),bits_list,True,group_size)


        layers = model.model.layers

        for i in tqdm(range(len(layers)), desc="(Weight Kurtosis)"):
            layer = layers[i]
            layer_key="layer: " + str(i)

            
            weights_layer_list = make_model_kurtosis_aggregator(group_size)
            
            for name, param in layer.named_parameters():
                if (len(param.data.shape)) !=2 :
                    continue
                layer_prefix = "layers."+str(i)
                
                tensor_data = param.data.detach().cpu().numpy()
                cleaned_name = layer_prefix+name.replace('.weight', '')

                # 이곳에서 Draw하는 부분을 Skip한다
                if int(i) in visualize_index and ptq_args.draw:
                    print(f"Visualizing 3D surface for layer: {name}...")
                    layer_visualize_path = os.path.join(weights_visualize_path, f"layer_{int(i)}")
                    os.makedirs(layer_visualize_path, exist_ok=True)

                    # 텐서를 numpy 배열로 변환
                    
                    
                    # X, Y 축에 사용할 좌표 생성
                    x_coords = np.arange(tensor_data.shape[0]) # out dim
                    y_coords = np.arange(tensor_data.shape[1]) # input dim
                    
                    # 2차원 그리드 생성
                    X, Y = np.meshgrid(x_coords, y_coords)
                    
                    # 3D 플롯 생성
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # Z축 값으로 텐서 데이터를 사용하여 3D 표면 그래프 그리기
                    ax.plot_surface(X, Y, tensor_data.T, cmap='coolwarm' )
                    
                    z_min = -0.1
                    z_max = tensor_data.max()
                    ax.set_zlim(z_min, z_max)

                    # 축 라벨과 제목 설정
                    ax.set_xlabel('channel')
                    

                    # ax.set_ylabel('Input Dim')
                    # ax.set_zlabel('Weight Value')
                    

                    
                    ax.set_title(f'{cleaned_name}')
                    # 플롯 보여주기
                    fig_path = os.path.join(layer_visualize_path, f"{cleaned_name}.png")
                    plt.savefig(fig_path)
                    print(f"3D plot saved to: {fig_path}")
                    plt.close(fig)

                # average list를 통하여 정확한 값을 
                weights_tensor[cleaned_name]=compute_stats_for_matrix(tensor_data,bits_list,True,group_size)
                row_sum,row_cnt=aggregate_kurtosis(tensor_data)

                weights_layer_list['per-row']['sum']+=row_sum
                weights_layer_list['per-row']['cnt']+=row_cnt
                for g_size in group_size:
                    group_key = f"group_size {g_size}"
                    grouped_data = safe_group_view(tensor_data,g_size)
                    g_sum,g_cnt = aggregate_kurtosis(grouped_data)
                    weights_layer_list[group_key]['sum']+=g_sum
                    weights_layer_list[group_key]['cnt']+=g_cnt
            
            weights_layer[layer_key]['per-row']['kurtosis']=(weights_layer_list['per-row']['sum']/weights_layer_list['per-row']['cnt'])
            for g_size in group_size:
                group_key = f"group_size {g_size}"
                weights_layer[layer_key][group_key]['kurtosis']=(weights_layer_list[group_key]['sum']/weights_layer_list[group_key]['cnt'])


      
        weight_tensor_path = os.path.join(dist_path,"weights_tensor.json")


        with open(weight_tensor_path,"w") as f:
            json.dump(weights_tensor,f,indent=4) 

        weight_layer_path = os.path.join(dist_path,"weights_layer.json")
        with open(weight_layer_path,"w") as f:
            json.dump(weights_layer,f,indent=4) 


    #         weights를 저장할 File을 만들고 거기다가 Flush 한다
    #         Weight의 Distribution을 저장하는 함수를 구현



    # Step 2: Activation의 Distribution을 저장을 한다
    # model.cuda()
    # layers=model.model.layers
    # for i in tqdm(range(len(layers)), desc="(Check) activations"):
    #     layer = layers[i]
    #     capture_layer_io = model_utils.capture_layer_io()

    

    if ptq_args.capture_layer_io:
       
        model.cuda()
        def hook_factory(module_name, captured_vals, is_input):
            def hook(module, input, output):
                if is_input:
                    captured_vals[module_name].append(input[0].detach().cpu())
                else:
                    captured_vals[module_name].append(output.detach().cpu())

            return hook

        if ptq_args.draw:
            acts_visualize_path = os.path.join(visualize_path,"Acts")
            os.makedirs(acts_visualize_path,exist_ok=True)

        activation_path = os.path.join(save_path, "Acts")
        os.makedirs(activation_path, exist_ok=True)

        
        

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

    
        for i in tqdm(range(len(layers)), desc="(Capture Activation)"):
            layer = layers[i].cuda()

            ### Activation의 Data를 write한다
            handles = []
            captured_inputs = {
            "k_proj": [],  # q_proj, v_proj has the same input as k_proj
            "o_proj": [],
            "gate_proj": [],  # up_proj has the same input as gate_proj
            "down_proj": [],
            }

            captured_outputs = {
                "v_proj": [],
            }

            for name in captured_inputs.keys():
                module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
                if (module is None):
                    print(name)
                    return
                handles.append(module.register_forward_hook(hook_factory(name,captured_inputs,True)))
            
            for name in captured_outputs.keys():
                module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
                handles.append(module.register_forward_hook(hook_factory(name,captured_outputs,False)))

            outs=layer(inps,attention_mask=attention_mask,position_ids=position_ids)[0]

            for module_name in captured_inputs:
                cat_tensor = torch.cat(captured_inputs[module_name],dim=0)
                captured_inputs[module_name] = cat_tensor.reshape(-1,cat_tensor.shape[-1])
            for module_name in captured_outputs:
                cat_tensor = torch.cat(captured_outputs[module_name],dim=0)
                captured_outputs[module_name] = cat_tensor.reshape(-1,cat_tensor.shape[-1])

            captured_io={"input":captured_inputs,"outputs":captured_outputs}
            layer_act_path = os.path.join(activation_path,f"layer:{i:03d}.pt")
            torch.save(captured_io,layer_act_path)

            for h in handles:
                h.remove()
            

            
            layers[i]=layer.cpu()
            del layer
            torch.cuda.empty_cache()

            inps = outs
            del outs
            
            # layers[i] = layers[i].cpu()
            ## Activation그림을 그리자

            if ptq_args.draw and i in visualize_index:

                layer_visualize_path=os.path.join(acts_visualize_path,f"layer_{i}")
                os.makedirs(layer_visualize_path,exist_ok=True)

                for key,items in captured_inputs.items():
                    tensor_data = items.cpu().numpy()
                    x_coords = np.arange(tensor_data.shape[1]) # input dimension 축으로 진행됨
                    y_coords = np.arange(tensor_data.shape[0]) # seq_len * batch 축으로 진행됨

                    X,Y = np.meshgrid(x_coords,y_coords)
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')

                    ax.plot_surface(X,Y,tensor_data,cmap='coolwarm')

                    z_min=tensor_data.min()
                    z_max = tensor_data.max()
                    ax.set_zlim(z_min, z_max)
                    
                    ax.set_xlabel('channel')

                    cleaned_name= 'layers.'+str(i)
                    if key in ["k_proj","o_proj","v_proj","q_proj"]:
                        cleaned_name+=".self_attn."+key
                    else:
                        cleaned_name +=".mlp."+key

                    ax.set_title(f'{cleaned_name}')
                    fig_path = os.path.join(layer_visualize_path, f"{cleaned_name}_input.png")
                    plt.savefig(fig_path)
                    print(f"3D plot saved to: {fig_path}")
                    plt.close(fig)

                for key,items in captured_outputs.items():
                    tensor_data = items.cpu().numpy()
                    x_coords = np.arange(tensor_data.shape[1]) # input dimension 축으로 진행됨
                    y_coords = np.arange(tensor_data.shape[0]) # seq_len * batch 축으로 진행됨

                    X,Y = np.meshgrid(x_coords,y_coords)
                    fig = plt.figure()
                    ax = fig.add_subplot(111,projection='3d')

                    ax.plot_surface(X,Y,tensor_data,cmap='coolwarm')

                    z_min=-0.1
                    z_max = tensor_data.max()

                    ax.set_xlabel('channel')

                    cleaned_name= 'layers.'+str(i)
                    if key in ["k_proj","o_proj","v_proj","q_proj"]:
                        cleaned_name+=".self_attn."+key
                    else:
                        cleaned_name +=".mlp."+key

                    ax.set_title(f'{cleaned_name}')
                    fig_path = os.path.join(layer_visualize_path, f"{cleaned_name}_output.png")
                    plt.savefig(fig_path)
                    print(f"3D plot saved to: {fig_path}")
                    plt.close(fig)

           
        #
        #     activation_stats = {}
        
        # activation_stats[i] = {"inputs": [],"outputs": []}

        # for key,items in captured_inputs.items():


    # # Step 4: Evaluation을 실제로 적용한다 # 

    # model.cuda()
    # inputs=inputs.to(model.device)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    # Activation_stats_num=0
    # for k,v in activation_stats.items():
    #     Activation_stats_num+=1
    # print(f"Collected activation statistics for {Activation_stats_num} components")

    # # Step 5: 관련된 Activation 함수를 저장한다 
    # act_path = save_path+"activation.json"
    # with open(act_path,"w") as f:
    #         json.dump(activation_stats,f,indent=4) 

if __name__ == "__main__":
    train()
