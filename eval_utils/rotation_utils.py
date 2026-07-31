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


def rotate_embeddings(model, R1: torch.Tensor, args,model_args) -> None:
    # Rotate the embeddings.
    # print("Rotate Embeddings")
    for W in [model.model.embed_tokens]:
        if args.diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=False,Matrix=R1) # W @ R1 
        else:
            dtype = W.weight.data.dtype
            dev=W.weight.device

            if args.offload:
                # CPU Offloading 모드 (VRAM 절약형)
                calc_device = "cpu"
                calc_dtype = torch.float64
            else:
                # GPU Direct 모드 (속도 최적화형)
                calc_device = dev
                # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
                calc_dtype = torch.float32

            # dev = R1.device
            W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
            W.weight.data = torch.matmul(W_, R1.to(device=calc_device,dtype=calc_dtype)).to(device=dev, dtype=dtype) # 기존이랑 다르게 Rotation을 적용해야 하는 것으로 보임
            if calc_device != "cpu":
                torch.cuda.empty_cache()
            # R1.to(dev)

def rotate_attention_inputs(layer, R1, args) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    # print("Attention Inputs")
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        if args.diagonal:
            apply_exact_had_to_linear(W,had_dim = R1.shape[1],Dim0=False,Matrix=R1) # W @ R1
        else: 
            dtype = W.weight.dtype
            dev = W.weight.device

            if args.offload:
                # CPU Offloading 모드 (VRAM 절약형)
                calc_device = "cpu"
                calc_dtype = torch.float64
            else:
                # GPU Direct 모드 (속도 최적화형)
                calc_device = dev
                # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
                calc_dtype = torch.float32

            W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
            W.weight.data = torch.matmul(W_, R1.to(device=calc_device,dtype=calc_dtype)).to(device=dev, dtype=dtype) # 기존이랑 다르게 Rotation을 적용해야 하는 것으로 보임
            if calc_device != "cpu":
                torch.cuda.empty_cache()


def rotate_attention_output(layer, R1, args) -> None:
    # Rotate output matrix of the self-attention layer.
    # print("Attention Output")
    W = layer.self_attn.o_proj
    if args.diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=True,Matrix=R1) # (W.T@R1)T
    else:
        dtype = W.weight.data.dtype
        dev = W.weight.device
        if args.offload:
            # CPU Offloading 모드 (VRAM 절약형)
            calc_device = "cpu"
            calc_dtype = torch.float64
        else:
            # GPU Direct 모드 (속도 최적화형)
            calc_device = dev
            # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
            calc_dtype = torch.float32
        W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
        W.weight.data = torch.matmul(R1.to(device=calc_device,dtype=calc_dtype).T, W_).to(device=dev, dtype=dtype)
        
    if W.bias is not None:
        b = W.bias.data.to(device=calc_device, dtype=calc_dtype)
        dev = W.weight.device
        W.bias.data = torch.matmul(R1.to(device=calc_device,dtype=calc_dtype).T, b).to(device=dev, dtype=dtype)


def rotate_mlp_input(layer, R1,args):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    # print("MLP Inputs")
    for W in mlp_inputs:
        if args.diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=False,Matrix=R1)
        else: 
            dtype = W.weight.dtype
            dev = W.weight.device

            if args.offload:
                # CPU Offloading 모드 (VRAM 절약형)
                calc_device = "cpu"
                calc_dtype = torch.float64
            else:
                # GPU Direct 모드 (속도 최적화형)
                calc_device = dev
                # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
                calc_dtype = torch.float32
            W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
            W.weight.data = torch.matmul(W_,R1.to(device=calc_device,dtype=calc_dtype)).to(device=dev, dtype=dtype)


def rotate_mlp_output(layer, R1, args):
    # Rotate the MLP output weights and bias.
    # print("MLP outputs")
    W = layer.mlp.down_proj
    if args.diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=True,Matrix=R1) # (W1.T @ R1)T => R1.T @ W1
    else:
        dtype = W.weight.data.dtype
        dev = W.weight.device

        if args.offload:
            # CPU Offloading 모드 (VRAM 절약형)
            calc_device = "cpu"
            calc_dtype = torch.float64
        else:
            # GPU Direct 모드 (속도 최적화형)
            calc_device = dev
            # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
            calc_dtype = torch.float32

        W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
        W.weight.data = torch.matmul(R1.to(device=calc_device,dtype=calc_dtype).T, W_).to(device=dev, dtype=dtype)
        
        if W.bias is not None:
            b = W.bias.data.to(device=calc_device, dtype=calc_dtype)
            dev = W.weight.device
            W.bias.data = torch.matmul(R1.to(device=calc_device,dtype=calc_dtype).T, b).to(device=dev, dtype=dtype)


def rotate_head(model, R1: torch.Tensor,args) -> None:
    # Rotate the head.
    # print("LM HEAD")
    W = model.lm_head
    if args.diagonal:
            apply_exact_had_to_linear(W,had_dim=R1.shape[1],Dim0=False,Matrix=R1)
    else:
        dtype = W.weight.data.dtype
        dev = W.weight.device
        if args.offload:
            # CPU Offloading 모드 (VRAM 절약형)
            calc_device = "cpu"
            calc_dtype = torch.float64
        else:
            # GPU Direct 모드 (속도 최적화형)
            calc_device = dev
            # GPU에서는 float64보다 float32가 훨씬 빠릅니다. 정밀도가 아주 민감하지 않다면 f32 권장
            calc_dtype = torch.float32

        W_ = W.weight.data.to(device=calc_device, dtype=calc_dtype)
        W.weight.data = torch.matmul(W_, R1.to(device=calc_device,dtype=calc_dtype)).to(device=dev, dtype=dtype) # 기존이랑 다르게 Rotation을 적용해야 하는 것으로 보임

        if calc_device != "cpu":
            torch.cuda.empty_cache()


def rotate_ov_proj(layer, head_num, head_dim, R2=None,online_r2=False):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    linear_dtype = v_proj.weight.dtype
    linear_device = v_proj.weight.device
    # print("OV_proj")
    
    # QuaRot 방식과 동일하게 R2 방식을 적용하면 diagonal하게 Randomized한 Hadamard rotation을 적용할 수가 없다 => 
    if (online_r2):
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, Dim0=True, Matrix=None)
        apply_exact_had_to_linear(o_proj, had_dim=-1, Dim0=False, Matrix=None)

        if hasattr(v_proj,"bias"):
            if v_proj.bias is not None: #Qwen2의 architecture 기반의 case
                original_shape = v_proj.bias.data.shape
                bias_reshaped = v_proj.bias.data.reshape(-1,head_dim)
                v_proj.bias.data = HadamardTransform.apply(bias_reshaped.float()/math.sqrt(head_dim)).to(dtype=linear_dtype).reshape(original_shape)
    else:
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, Dim0=True, Matrix=R2)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, Dim0=False, Matrix=R2)

        if hasattr(v_proj,"bias"):
            if v_proj.bias is not None: #Qwen2의 architecture 기반의 case v_proj의 bias가 포함되어 있음
                original_shape = v_proj.bias.data.shape
                bias_reshaped = v_proj.bias.data.reshape(-1,head_dim) # (dim0//head_dim,head_dim) @ (head_dim,head_dim)
                v_proj.bias.data = torch.matmul(bias_reshaped.to(dtype=torch.float32),R2.to(dtype=torch.float32,device=linear_device)).to(dtype=linear_dtype).reshape(original_shape)

def compute_residual_subspace(T: torch.Tensor, rank: int):
    """ReSpinQuant subspace residual rotation approximation (paper Sec. 3.3).

    Given a residual transition matrix T = R_in^T R_out (D x D, applied as x @ T),
    approximate it by a rank-r rotation confined to the principal mismatch subspace:
        T_hat = I + Q (R_sub - I_r) Q^T
    so that  x @ T_hat = x + ((x @ Q) @ M) @ Q^T,  with  M = R_sub - I_r.

    Returns (Q [D, r], M [r, r]) in float32.

    (The exact full-rank case, rank <= 0 or rank >= D, is handled by the caller,
    which stores the full transition T directly instead of a Q/M factorization.)
    """
    T = T.to(torch.float64)
    D = T.shape[0]
    r = min(rank, D)
    I = torch.eye(D, dtype=T.dtype, device=T.device)
    # (5) principal directions of the deviation Delta_T = T - I
    U, _, _ = torch.linalg.svd(T - I)
    Q = U[:, :r]                          # D x r
    # (6) project T onto the subspace
    T_sub = Q.T @ T @ Q                   # r x r
    # (7)-(8) closest orthogonal matrix via polar decomposition (SVD)
    Us, _, Vhs = torch.linalg.svd(T_sub)
    R_sub = Us @ Vhs                      # r x r, in SO(r)
    M = R_sub - torch.eye(r, dtype=T.dtype, device=T.device)
    return Q.to(torch.float32).contiguous(), M.to(torch.float32).contiguous()


@torch.inference_mode()
def rotate_model_lierespinquant(model, args, model_args=None):
    """LieReSpinQuant offline fusion.

    Same weight fusion as :func:`rotate_model_respinquant`, but the bases are
    rebuilt from the learned Lie chain

        B_0 = Hadamard (fixed),   B_{k+1} = B_k @ dR_k,
        dR_k = Cayley(U_k diag(g_k) V_k^T - V_k diag(g_k) U_k^T) = I + P_k Z_k P_k^T

    so the residual transition ``B_k^T B_{k+1}`` *is* ``dR_k`` and is already in
    rank-2r factored form.  There is no SVD, no polar decomposition and no
    determinant correction: the installed correction is the exact learned
    rotation, not a truncated approximation of a dense one.

    The factors are stored in the same ``Q_attn``/``M_attn`` slots the eval
    forward already uses, since it computes ``residual + ((residual @ Q) @ M) @ Q^T``
    -- with ``Q = P`` and ``M = Z`` that is exactly ``residual @ dR``.
    """
    from utils.lie_rotation import bases_from_factors, cayley_lowrank_factors

    assert args.optimized_rotation_path is not None, (
        "LieReSpinQuant requires --optimized_rotation_path pointing to a trained R.bin"
    )
    ckpt = torch.load(args.optimized_rotation_path, map_location="cpu")
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    num_transitions = 2 * num_layers

    # Locate the chain regardless of the FSDP/DDP prefix the checkpoint was saved with.
    base_key = next(k for k in ckpt if k.endswith("lie_chain.base"))
    prefix = base_key[: -len("base")]
    base = ckpt[base_key].to(device="cuda", dtype=torch.float32)

    factors = []
    for k in range(num_transitions):
        U = ckpt[f"{prefix}deltas.{k}.U"].to(device="cuda", dtype=torch.float32)
        V = ckpt[f"{prefix}deltas.{k}.V"].to(device="cuda", dtype=torch.float32)
        g = ckpt[f"{prefix}deltas.{k}.gamma"].to(device="cuda", dtype=torch.float32)
        factors.append(cayley_lowrank_factors(U, V, g))
    bases = bases_from_factors(base, factors)   # 2L+1 tensors, D x D
    assert len(bases) == num_transitions + 1

    def _head_R2(idx):
        key = next(k for k in ckpt if k.endswith(f"layers.{idx}.self_attn.R2"))
        return ckpt[key].to(device="cuda", dtype=torch.float32)

    if not args.deactivate_r1:
        rotate_embeddings(model, bases[0], args, model_args)
        rotate_head(model, bases[-1], args)
        utils.cleanup_memory()

    layers = list(model.model.layers)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer",
                                          desc="Rotating (LieReSpinQuant)")):
        if not args.deactivate_r2:
            rotate_ov_proj(layer, num_heads, head_dim, R2=_head_R2(idx), online_r2=False)

        if not args.deactivate_r1:
            R1_in = bases[2 * idx]
            R2_mid = bases[2 * idx + 1]
            R1_next = bases[2 * idx + 2]

            rotate_attention_inputs(layer, R1_in, args)      # q/k/v: W @ R1_in
            rotate_attention_output(layer, R2_mid, args)     # o_proj: R2_mid^T @ W
            rotate_mlp_input(layer, R2_mid, args)            # gate/up: W @ R2_mid
            rotate_mlp_output(layer, R1_next, args)          # down: R1_next^T @ W

            dev = layer.self_attn.o_proj.weight.device
            if getattr(args, "deactivate_residual", False):
                pass  # A/B switch: leave the basis mismatch uncorrected
            else:
                P_attn, Z_attn = factors[2 * idx]
                P_ffn, Z_ffn = factors[2 * idx + 1]
                # residual + ((residual @ P) @ Z) @ P^T  ==  residual @ dR, exactly
                layer.Q_attn = P_attn.to(dev).contiguous()
                layer.M_attn = Z_attn.to(dev).contiguous()
                layer.Q_ffn = P_ffn.to(dev).contiguous()
                layer.M_ffn = Z_ffn.to(dev).contiguous()
            torch.cuda.empty_cache()

    del bases, factors
    utils.cleanup_memory()


@torch.inference_mode()
def rotate_model_respinquant(model, args, model_args=None):
    """ReSpinQuant offline fusion: fuse DISTINCT per-layer R1/R2 into each layer's
    weights (as in global rotation), and install a low-rank subspace correction on
    each residual connection to resolve the resulting basis mismatch."""
    assert args.optimized_rotation_path is not None, (
        "ReSpinQuant requires --optimized_rotation_path pointing to a trained R.bin"
    )
    # Keep the checkpoint on CPU; move only the matrices we currently need to GPU.
    ckpt = torch.load(args.optimized_rotation_path, map_location="cpu")
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    rank = getattr(args, "residual_rank", 32)

    # float32 is enough: the GPU fusion path downcasts to float32 anyway, so this
    # yields byte-identical fused weights while halving the transient VRAM vs float64.
    def _load(key):
        return ckpt[key].to(device="cuda", dtype=torch.float32)

    # Embedding uses layer 0's input basis; lm_head un-rotates the final basis.
    if not args.deactivate_r1:
        R1_0 = _load("model.layers.0.R1")
        rotate_embeddings(model, R1_0, args, model_args)
        del R1_0
        R1_final_head = _load("model.R1_final")
        rotate_head(model, R1_final_head, args)
        del R1_final_head
        utils.cleanup_memory()

    layers = list(model.model.layers)
    # Rolling window: R1_next of layer i is R1_in of layer i+1, so we carry a single
    # R1 matrix across iterations instead of holding all L on the GPU at once.
    R1_in = _load("model.layers.0.R1") if not args.deactivate_r1 else None
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating (ReSpinQuant)")):
        # Head rotation R3 (= SpinQuant's per-layer self_attn.R2)
        if not args.deactivate_r2:
            head_R2 = _load(f"model.layers.{idx}.self_attn.R2")
            rotate_ov_proj(layer, num_heads, head_dim, R2=head_R2, online_r2=False)
            del head_R2

        if not args.deactivate_r1:
            R2_mid = _load(f"model.layers.{idx}.R2")             # attn out / ffn in basis
            R1_next = (
                _load(f"model.layers.{idx + 1}.R1")
                if idx + 1 < num_layers
                else _load("model.R1_final")
            )

            rotate_attention_inputs(layer, R1_in, args)         # q/k/v: W @ R1_in
            rotate_attention_output(layer, R2_mid, args)        # o_proj: R2_mid^T @ W
            rotate_mlp_input(layer, R2_mid, args)               # gate/up: W @ R2_mid
            rotate_mlp_output(layer, R1_next, args)             # down: R1_next^T @ W

            # Residual subspace corrections for the two basis transitions:
            #   attn skip: R1_in -> R2_mid ;  ffn skip: R2_mid -> R1_next
            dev = layer.self_attn.o_proj.weight.device
            T_attn = R1_in.T @ R2_mid
            T_ffn = R2_mid.T @ R1_next
            if getattr(args, "deactivate_residual", False):
                # A/B switch: leave the basis mismatch uncorrected (Q/M/T stay None).
                layer.T_attn = T_attn.to(device=dev, dtype=torch.float32).contiguous()
                layer.T_ffn = T_ffn.to(device=dev, dtype=torch.float32).contiguous()
                del R2_mid, R1_in
            else:
                
                D = T_attn.shape[0]
                if rank <= 0 or rank >= D:
                    # Exact full-rank correction: store T directly (forward: residual @ T).
                    # No SVD; lossless, so 16-bit PPL must match the un-rotated baseline.
                    layer.T_attn = T_attn.to(device=dev, dtype=torch.float32).contiguous()
                    layer.T_ffn = T_ffn.to(device=dev, dtype=torch.float32).contiguous()
                else:
                    Q_attn, M_attn = compute_residual_subspace(T_attn, rank)
                    Q_ffn, M_ffn = compute_residual_subspace(T_ffn, rank)
                    layer.Q_attn = Q_attn.to(dev)
                    layer.M_attn = M_attn.to(dev)
                    layer.Q_ffn = Q_ffn.to(dev)
                    layer.M_ffn = M_ffn.to(dev)
                del T_attn, T_ffn, R2_mid, R1_in
            R1_in = R1_next  # carry forward as next layer's attn-input basis
            torch.cuda.empty_cache()

    del R1_in
    utils.cleanup_memory()


@torch.inference_mode()
def rotate_model(model, args,model_args=None):

    if args.diagonal: 
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
    if (not args.deactivate_r1):
        rotate_embeddings(model,R1,args,model_args) 
        rotate_head(model,R1,args)

    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.diagonal:
            if args.diagonal_size > head_dim:
                R2 = get_orthogonal_matrix(head_dim,args.rotate_mode)
                if (not args.deactivate_r2):
                    rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2,online_r2=args.online_r2)
            else:
                R2 = get_orthogonal_matrix(args.diagonal_size,args.rotate_mode)
                if (not args.deactivate_r2):
                    rotate_ov_proj(layers[idx], num_heads, args.diagonal_size, R2=R2,online_r2=args.online_r2)
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
            online_r2 = args.online_r2
            if args.optimized_rotation_path is not None:
                key = f"model.layers.{idx}.self_attn.R2"
                R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
                online_r2=False

            if (not args.deactivate_r2):
                rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2,online_r2=online_r2)    
                
        if (not args.deactivate_r1):
            rotate_attention_inputs(layers[idx], R1, args)
            rotate_attention_output(layers[idx], R1, args)
            rotate_mlp_input(layers[idx], R1, args)
            rotate_mlp_output(layers[idx], R1, args)
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
