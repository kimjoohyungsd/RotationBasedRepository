# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import tqdm

from utils import quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    get_hadK
)
from utils.utils import HadamardTransform

from scipy.linalg import hadamard
import numpy as np


def R4_rotate_down_proj_weights(layer):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    apply_exact_had_to_linear(
        W, had_dim=-1, output=False
    )  # apply exact (inverse) hadamard on the weights of mlp output


@torch.inference_mode()
def rotate_model(model, args):
    # # Step1: 해당에 맞는 Config를 Setting 한다

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    hidden_size = config.intermediate_size

    # had_K, K = get_hadK(hidden_size)
    # n = hidden_size // K

    # # Step2 R4의 Inverse Matrix를 Manually 하게 구한다 
    # first_layer = model.model.layers[0]
    # dtype = first_layer.mlp.down_proj.weight.dtype
    # R4_matrix_np = np.kron(had_K.T.detach().cpu().numpy(), hadamard(n))
    # R4_matrix = torch.from_numpy(R4_matrix_np).to(dtype=dtype)
    # R4_matrix = R4_matrix * 1/torch.tensor(hidden_size, dtype=dtype).sqrt()

    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(
        tqdm.tqdm(layers, unit="layer", desc="Applying R4 rotation to W_down")
    ):
    #     w_ = layer.mlp.down_proj.weight.data
    #     dev = w_.device
    #     init_shape = w_.shape
        
    #     print(f"  Layer {idx} - Weight device: {dev}")
    #     print(f"  Layer {idx} - Weight dtype: {dtype}")
    #     print(f"  Layer {idx} - Weight shape: {w_.shape}")
        
    #     # Move both tensors to CPU for matmul (to avoid multi-GPU issues)
    #     # weight_cpu = original_down_weight.detach().cpu().to(dtype=original_dtype)
    #     # r4_cpu = R4_matrix.detach().cpu().to(dtype=original_dtype)
        
    #     # Perform matrix multiplication on CPU

    #     new_down_weight = torch.matmul(w_, R4_matrix.to(dev))
        
        
    #     # Assign as Parameter (CRITICAL: must be Parameter, not regular tensor)
    #     layer.mlp.down_proj.weight.data = new_down_weight.to(device=dev,dtype=dtype)
        
    #     print(f"  Layer {idx} - New weight device: {layer.mlp.down_proj.weight.device}")
    #     print(f"  Layer {idx} - New weight dtype: {layer.mlp.down_proj.weight.dtype}")

    #     print("R4 matrix fusion completed successfully!")
        
        R4_rotate_down_proj_weights(layers[idx])


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
    import functools

    from utils import monkeypatch

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
