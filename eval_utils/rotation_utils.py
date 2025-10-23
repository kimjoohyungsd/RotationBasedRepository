# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor, diagonal) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        if diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[0],Dim0=False,Matrix=R1) # W @ R1 
        else:
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype) # 기존이랑 다르게 Rotation을 적용해야 하는 것으로 보임


def rotate_attention_inputs(layer, R1, diagonal) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        if diagonal:
            apply_exact_had_to_linear(W,had_dim = R1.shape[0],Dim0=False,Matrix=R1) # W @ R1
        else: 
            dtype = W.weight.dtype
            W_ = W.weight.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1, diagonal) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj
    if diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[0],Dim0=True,Matrix=R1) # (W.T@R1)T
    else:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
        
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1,diagonal):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        if diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=False,Matrix=R1)
        else: 
            dtype = W.weight.dtype
            W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
            W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, diagonal):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    if diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[0],Dim0=True,Matrix=R1) # (W1.T @ R1)T => R1.T @ W1
    else:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    # apply_exact_had_to_linear(
    #     W, had_dim=-1, output=False,transpose=True
    # )  # apply exact (inverse) hadamard on the weights of mlp output (Hadamard Matrix를 정확하게 구현하자)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor,diagonal) -> None:
    # Rotate the head.
    W = model.lm_head
    if diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[0],Dim0=False,Matrix=R1)
    else:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None,online_r2=False):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    
    # QuaRot 방식과 동일하게 R2 방식을 적용하면 diagonal하게 Randomized한 Hadamard rotation을 적용할 수가 없다 => 
    if (online_r2):
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, Dim0=True, Matrix=None)
        apply_exact_had_to_linear(o_proj, had_dim=-1, Dim0=False, Matrix=None)
    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, Dim0=True, Matrix=R2)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, Dim0=False, Matrix=R2)


@torch.inference_mode()
def rotate_model(model, args):
    if args.diagonal: # R1도 Diagonal 하게 하는 경우 위와 같이 하나의 Diagonal size에 맞게 Rotation을 한다
        R1 = get_orthogonal_matrix(args.diagonal_size,args.rotate_mode)
    else:
        R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    # Rotation을 함에 있어서도 Diagonal 한 특성을 고려해서 Rotation을 진행한
    rotate_embeddings(model,R1,args.diagonal) 
    rotate_head(model,R1,args.diagonal)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        
        if args.diagonal:
            if args.diagonal_size > head_dim:
                R2 = get_orthogonal_matrix(head_dim,args.rotate_mode)
                rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2,online_r2=args.online_r2)
            else:
                R2 = get_orthogonal_matrix(args.diagonal_size,args.rotate_mode)
                rotate_ov_proj(layers[idx], num_heads, args.diagonal_size, R2=R2,online_r2=args.online_r2)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
            if args.optimized_rotation_path is not None:
                key = f"model.layers.{idx}.self_attn.R2"
                R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
            rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2,online_r2=args.online_r2)

        rotate_attention_inputs(layers[idx], R1, args.diagonal)
        rotate_attention_output(layers[idx], R1, args.diagonal)
        rotate_mlp_input(layers[idx], R1, args.diagonal)
        rotate_mlp_output(layers[idx], R1, args.diagonal)
        # rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs), # QkrotationWrapper의 forward 함수 kwargs dictionary에는 
    )
    setattr(module, attr_name, wrapper) # 복원을 위한 원래 rotary pos_ embedding 함수
