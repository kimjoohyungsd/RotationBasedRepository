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
import transformers

from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.utils import HadamardTransform, cleanup_memory


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
    """
    A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
    for the activations.
    """

    def __init__(self,percolumn) -> None:
        super(ActQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(1))
        self.register_buffer("zero", torch.zeros(1))
        self.bits = 16
        self.percolumn = percolumn
    def free(self) -> None:
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(
        self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0
    ) -> None:
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert (
            self.clip_ratio <= 1 and self.clip_ratio > 0
        ), "Clip ratio should be in (0, 1]"

    def find_params_per_token_groupwise(self, x) -> None:
        init_shape = x.shape
        reshaped_x = x.reshape(
            -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
        ) # reshaped_x [batch,seq,group_num,groupsize]

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio # [batch , Seq, group_num] * 
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x) -> None:
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))# [BATCH*SEQ, HIDDEN_DIM]

        # --- 수정 포인트 시작 ---
        if self.percolumn:
            # Column 단위: dim=0 (Token 방향으로 스캔하여 각 Channel의 통계를 구함)
            dim_to_reduce = 0
            repeat_shape = (reshaped_x.shape[0], 1) # [Tokens, 1] 방향으로 확장
            unsqueeze_dim = 0
        else:
            # Per-token(Row) 단위: dim=1 (Channel 방향으로 스캔하여 각 Token의 통계를 구함)
            dim_to_reduce = 1
            repeat_shape = (1, reshaped_x.shape[-1]) # [1, Hidden] 방향으로 확장
            unsqueeze_dim = 1

        # 통계값 추출
        # xmin/xmax의 Shape: percolumn ? [HIDDEN_DIM] : [BATCH*SEQ]
        tmp_zeros = torch.zeros(reshaped_x.shape[1] if self.percolumn else reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(dim_to_reduce)[0], tmp_zeros) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(dim_to_reduce)[0], tmp_zeros) * self.clip_ratio
        # --- 수정 포인트 끝 ---

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax) 
            invalid_mask = xmax == 0
            
            # Scale 계산 및 확장
            # percolumn이면 [1, HIDDEN_DIM]로 만든 후 Token 방향(repeat_shape[0])으로 복사
            # per-token이면 [BATCH*SEQ, 1]로 만든 후 Hidden 방향(repeat_shape[1])으로 복사
            scale = (xmax / self.maxq).unsqueeze(unsqueeze_dim).repeat(repeat_shape)
            scale[invalid_mask.unsqueeze(unsqueeze_dim).repeat(repeat_shape)] = 1
            
            self.scale = scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            # 비대칭(Asymmetric) 로직도 동일한 방식으로 적용
            invalid_mask = (xmin == 0) & (xmax == 0)
            xmin[invalid_mask] = -1
            xmax[invalid_mask] = +1
            
            scale = (xmax - xmin) / self.maxq
            zero = torch.round(-xmin / scale)
            
            self.scale = scale.unsqueeze(unsqueeze_dim).repeat(repeat_shape).reshape(init_shape)
            self.zero = zero.unsqueeze(unsqueeze_dim).repeat(repeat_shape).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    """
    This class is a wrapper for the activation quantization.
    We extract the FP features in the forward pass and quantize the rest using
    the self.quantizer object.
    If a rotation Q is provided, the weight matrix will be rotated,
    a pre-forward hook will be registered to rotate the activation before quantization.
    """

    def __init__(self, module: torch.nn.Linear,percolumn) -> None:
        super(ActQuantWrapper, self).__init__()
        # assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer(percolumn)
        self.out_quantizer = ActQuantizer(percolumn)
        self.register_buffer("had_K", torch.tensor(0))
        self._buffers["had_K"] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_diagonal_had = False
        self.had_dim = 0
        self.fp32_had = False
        # self.transpose = False

    def extra_repr(self) -> str:
        str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
        if self.quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
        if self.out_quantizer.bits < 16:
            str_ += (
                f" (Asymmetric Per-Token)"
                if not self.out_quantizer.sym
                else f" (Symmetric Per-Token)"
            )

        return str_

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                # print("Full Hadamard Matrix")
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K,transpose).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K,transpose)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            # print("Apply online_partial hadamard")
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim # [batch,Dim_group,had_dim] => [batch,had_dim,Dim_group]
                        ).transpose(1, 2)
                    )
                    / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (
                    self.had_K.to(x.dtype)
                    @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
                ) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)
        
        elif self.online_diagonal_had:
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = (
                    HadamardTransform.apply(
                        x.reshape(
                            -1, init_shape[-1] // self.had_dim, self.had_dim # [batch,Dim_group,had_dim] => [batch,had_dim,Dim_group]
                        )
                    )
                    / math.sqrt(self.had_dim)
                )

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()
        if R1 is not None:
            x = self.module(x, R1, R2, transpose).to(x_dtype)
        else:
            x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape: int = 1) -> None:
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        percolumn:  bool = False,
        perchannel: bool = False,
        sym: bool = True,
        mse: bool = False,
        norm: float = 2.4,
        grid: int = 100,
        maxshrink: float = 0.8,
        weight_groupsize: int = -1,
    ) -> None:
        self.bits = bits
        self.perchannel = perchannel
        self.percolumn = percolumn
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.weight_groupsize = weight_groupsize
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params_weight_groupwise(self, x) -> None:
        init_shape = x.shape
        x = x.reshape(
            x.shape[-2], x.shape[-1] // self.weight_groupsize, self.weight_groupsize
        ) # [Dim0,dim1/group_size,group_size]

        xmax = torch.amax(x, dim=-1, keepdim=True) # [Dim0, Dim1 / group_size,1]
        xmin = torch.amin(x, dim=-1, keepdim=True) # [Dim0, Dim1 / group_size,1]

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) # [Dim0,Dim1 / group,1]
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, self.weight_groupsize) # [Dim0,Dim1 / group_size , group_size]
        self.zero = self.zero.repeat(1, 1, self.weight_groupsize) # [Dim0, Dim1 / group_size, group_size]

        if self.mse:
            best = torch.full(
                [x.shape[0], x.shape[1]], float("inf"), device=x.device
            ).type_as(x)
            for i in range(int(self.maxshrink * self.grid)): # 여러 Grid: 격자점간의 간격, MaxShrink: 최대 몇 격자점 단위로 shrink 되는지
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = sym_quant_dequant(x, scale1, self.maxq) # dequant했을때의 값을 구한다
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    scale1 = scale1.repeat(1, 1, self.weight_groupsize)
                    zero1 = zero1.repeat(1, 1, self.weight_groupsize)
                    q = asym_quant_dequant(x, scale1, zero1, self.maxq)

                q -= x # [dim0, dim1 / group_size,group_size]
                q.abs_() # [dim0,dim1,group_size]
                q.pow_(self.norm)
                err = torch.sum(q, -1,dtype=best.dtype) #[dim0,dim1/group_size]
                tmp = err < best # [dim0 , dim1/group_size]
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        self.scale = self.scale.reshape(init_shape)
        self.zero = self.zero.reshape(init_shape)

    def find_params(self, x) -> None: # [out_dim,in_dim]
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape # [out_dim, in_dim]

        if self.weight_groupsize > 0:
            # group-wise per-token quantization
            self.find_params_weight_groupwise(x)
            cleanup_memory(verbos=False)
            return
        elif self.perchannel:
            x = x.flatten(1) # Perchannel인 경우
            if self.percolumn: # Column 단위로 Quantization을 하는 경우
                x = x.transpose(1,0) 
        else:
            x = x.flatten().unsqueeze(0) # 


        tmp = torch.zeros(x.shape[0], device=dev) # per_channel인 경우 [out_dim] per-column을 적용한 경우 [in_dim]
        xmin = torch.minimum(x.min(1)[0], tmp)  # [out_dim]  per-column을 적용한 경우 [in_dim]
        xmax = torch.maximum(x.max(1)[0], tmp)  # [out_dim]  per-column을 적용한 경우 [in_dim]

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq #
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)): # grid: 한 격자점의 간격, Maxshrink: 최대 멀어지는 간격의 크기
                p = 1 - i / self.grid
                xmin1 = p * xmin # [out_dim] , per-column [in_dim]
                xmax1 = p * xmax # [out_dim], per-column [in_dim]

                if self.sym: # []
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        # if self.percolumn:
        #     self.scale = self.scale.reshape(1,-1)
        #     self.zero = self.zero.reshape(1,-1)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):
        if self.percolumn:
            x=x.transpose(1,0)
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
            out = (scale * q).to(x_dtype)
            if self.percolumn:
                out = out.transpose(0, 1)
            return out, q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
    percolumn: bool=False
) -> None:
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module): # 입력인자로 들어온 module안에 모든 변수와 메소드를 가지고 와줌
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp,percolumn)) # 해당 모듈의 속성 값을 ActQuantWrapper로 설정한다
        if type(tmp) is torch.nn.Sequential: # 해당 멤버변수의 Type이 nn.Sequential인 경우
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child,percolumn))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) is torch.nn.ModuleList: # 
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child,percolumn))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
            
    for name1, child in module.named_children(): # 모듈의 바로 직전 output의 해당하는 
        add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
    # fix for llama embedding layer
    if type(module) in [torch.nn.Embedding] and type(module) in layers:
        return {"embed_tokens": module}
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children(): # named_children은 바로 밑에 있는 Tree구조를 가지고 실제 적용 
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res
