import torch
import torch.nn as nn
import functools

from tqdm import tqdm
from utils import data_utils


def get_massdiff_indices(scores, block_size):
    """
    scores [hid_dim]를 입력받아 각 블록이 균등한 절대값 질량(L1-norm)을 
    가지도록 분할하는 인덱스 맵 생성 (MassDiff)
    """
    hid_dim = scores.shape[0]
    if hid_dim <= block_size:
        return torch.arange(hid_dim)
    _, indexes = torch.sort(scores, descending=True) # 
    num_blocks = hid_dim // block_size

    # 각 블록의 누적 점수와 인덱스 리스트 초기화
    block_scores = scores[indexes[:num_blocks]].clone()
    block_idxs = [[idx.item()] for idx in indexes[:num_blocks]]

    for idx in indexes[num_blocks:]:
        # 가장 점수 누적이 적은 블록을 찾아 할당 (Greedy)
        min_block = torch.argmin(block_scores)
        block_scores[min_block] += scores[idx]
        block_idxs[min_block].append(idx.item())
        
        # 블록이 가득 차면 후보에서 제외하기 위해 무한대 부여
        if len(block_idxs[min_block]) == block_size:
            block_scores[min_block] = float('inf')

    final_indices = torch.tensor([idx for block in block_idxs for idx in block], dtype=torch.long)
    return final_indices

def get_zigzag_indices(scores,block_size):
    hidden_dim = scores.shape[-1]
    pairs = [(i, scores[i].item()) for i in range(hidden_dim)]
    pairs.sort(key=lambda x: x[1], reverse=True)
    def zigzag(numbers):
            cur = 0
            up = True
            l = [[] for i in range(hidden_dim // block_size)] # list1, list2, list3, .. list 32
            for i in range(len(numbers)): # list
                l[cur].append(numbers[i])
                if up:
                    cur += 1  # 
                    if cur == len(l):
                        cur -= 1
                        up = False
                else:
                    cur -= 1  #
                    if cur == -1:
                        cur += 1
                        up = True
            return l
    
    pairs = zigzag(pairs)

    for i in range(len(pairs)):
            pairs[i].sort(key=lambda x: x[1], reverse=True)
    perm = torch.zeros(hidden_dim, dtype=torch.long)
    for i in range(len(pairs)):
            perm[i * block_size:(i+1) * block_size] = torch.tensor([_[0] for _ in pairs[i]])

    return perm

def permute_random(model,args,tokenizer):
    num_layers = len(model.model.layers)
    layers = model.model.layers
    intermediate_dim = layers[0].mlp.gate_proj.out_features

    for i in tqdm(range(num_layers), desc='Random Permute'):
        layer=layers[i]
        mlp = layer.mlp
        dev = mlp.gate_proj.weight.device

        # 1. 섞기 위한 무작위 인덱스 맵 생성 (0 ~ intermediate_dim - 1)
        perm_indices = torch.randperm(intermediate_dim, device=dev)
        
        # 2. Gate Projection 섞기 (Output Channel 방향 셔플
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(mlp.gate_proj.weight[perm_indices, :])
            if mlp.gate_proj.bias is not None:
                mlp.gate_proj.bias.copy_(mlp.gate_proj.bias[perm_indices])

        # 3. Up Projection 섞기 (Output Channel 방향 셔플)
        with torch.no_grad():
            mlp.up_proj.weight.copy_(mlp.up_proj.weight[perm_indices, :])
            if mlp.up_proj.bias is not None:
                mlp.up_proj.bias.copy_(mlp.up_proj.bias[perm_indices])

        # 4. Down Projection 섞기 (Input Channel 방향 셔플)
        with torch.no_grad():
            mlp.down_proj.weight.copy_(mlp.down_proj.weight[:, perm_indices])

@torch.no_grad()
def permute_calibrate(model,args,tokenizer):
    dataloader = data_utils.get_wikitext2(tokenizer=tokenizer)
    nsamples = len(dataloader)
    device = next(model.parameters()).device
    
    #1. 통계값 도출하기
    
    act_scales = {}
    def stat_tensor(args,name,tensor):
        intermediate_dim=tensor.shape[-1]
        tensor = tensor.view(-1,intermediate_dim).abs().detach()

        if args.permute_mode=='massdiff':
            scores = torch.abs(tensor).mean(dim=0).cpu()
        elif args.permute_mode == 'zigzag':
            scores = torch.abs(tensor).max(dim=0).values.cpu()

        if name in act_scales:
            if args.permute_mode=='massdiff':
                act_scales[name] += scores
            elif args.permute_mode =='zigzag':
                act_scales[name] = torch.max(act_scales[name], scores)
        else:
            act_scales[name] = scores

    def stat_input_hook(m,x,y,name,args):
        if isinstance(x,tuple):
            x=x[0]
        stat_tensor(args,name,x)
    
    hooks = []
    for name,m in model.named_modules():
        if isinstance(m,nn.Linear) and 'down_proj' in name:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook,name=name,args=args)
                )
            )

    for i in tqdm(range(args.nsamples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    #2. 각각의 scales의 맞게 permute_index를 구함
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'down_proj' in name:

            device = m.weight.device
            scores = act_scales[name]
            if args.permute_mode == 'massdiff':
                perm_indices= get_massdiff_indices(scores, block_size=args.diagonal_size)
            elif args.permute_mode == 'zigzag':
                perm_indices = get_zigzag_indices(scores,block_size=args.diagonal_size)
            perm_indices = perm_indices.to(device)

            parent_path = name.rsplit('.', 1)[0] # model.layers.{layer_idx}.mlp_proj.down_proj
            mlp_module = dict(model.named_modules())[parent_path]
        
        # 2-3. 일치성을 깨뜨리지 않고 가중치 순서 바꾸기 (In-place copy)
            with torch.no_grad():
                # (1) gate_proj & up_proj: Output Channel 변경 [Out_dim, In_dim]
                mlp_module.gate_proj.weight.copy_(mlp_module.gate_proj.weight[perm_indices, :])
                if mlp_module.gate_proj.bias is not None:
                    mlp_module.gate_proj.bias.copy_(mlp_module.gate_proj.bias[perm_indices])
                    
                mlp_module.up_proj.weight.copy_(mlp_module.up_proj.weight[perm_indices, :])
                if mlp_module.up_proj.bias is not None:
                    mlp_module.up_proj.bias.copy_(mlp_module.up_proj.bias[perm_indices])
                    
                # (2) down_proj: Input Channel 변경 [Out_dim, In_dim] -> dim=1을 변경
                mlp_module.down_proj.weight.copy_(mlp_module.down_proj.weight[:, perm_indices])

    print("Permutation completed successfully!")
    # num_layers = len(model.model.layers)
    # layers = model.model.layers
    # intermediate_dim = layers[0].mlp.gate_proj.out_features


def permute(model,args,log,tokenizer):
    if args.permute_mode=='random':
        permute_random(model,args,tokenizer)
        log.info("Random Permutation complete")
    else:
        permute_calibrate(model,args,tokenizer)
        log.info("Calibrated Permutation {} complete".format(args.permute_mode))

