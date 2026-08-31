# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
import tqdm

from utils import quant_utils, utils


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        export_to_et=False,
    ):
        W = self.layer.weight.data.clone()
        W = W.float()
        Scale = self.layer.weight.data.clone()
        Scale = Scale.float()
        W_int = self.layer.weight.data.clone()
        W_int = W_int.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            W_int1 = torch.zeros_like(W1)
            Scale1 = torch.zeros_like(W1).to(Scale.dtype)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q, int_weight, scale = self.quantizer.fake_quantize(w.unsqueeze(1))
                Q1[:, i] = q.flatten()
                q = q.flatten()
                W_int1[:, i] = int_weight.flatten()
                Scale1[:, i] = scale.flatten()

                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W_int[:, i1:i2] = W_int1
            Scale[:, i1:i2] = Scale1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        if export_to_et:
            self.layer.register_buffer(
                "int_weight", W_int.reshape(self.layer.weight.shape)
            )
            self.layer.register_buffer("scale", Scale)
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # --distribute (device_map="auto") dispatches the model with accelerate: every
    # submodule carries an AlignDevicesHook that re-sends the *inputs* to the device the
    # module was assigned to. Moving a layer by hand (`layers[i].to(dev)`) therefore
    # desynchronizes weights (moved) from the hook's execution_device (unchanged) and
    # blows up with "found at least two devices, cuda:0 and cuda:1". In dispatched mode
    # we move NOTHING: each layer is quantized in place, on whatever device accelerate
    # gave it, and only the calibration tensors travel.
    device_map = getattr(model, "hf_device_map", None) or {}
    dispatched = len(device_map) > 0
    if dispatched:
        offloaded = sorted(
            k for k, v in device_map.items() if str(v) in ("cpu", "disk", "meta")
        )
        assert not offloaded, (
            "GPTQ cannot quantize accelerate-offloaded modules: their weights are "
            "streamed from a read-only weights_map, so in-place GPTQ updates would be "
            "silently discarded. Offloaded modules: {}...  Either raise --max_memory "
            "so the whole model fits on the GPUs, or drop --distribute and use "
            "--gptq_cpu_offload (gptq_fwrd_distribute).".format(offloaded[:5])
        )
        dev = model.model.embed_tokens.weight.device
        logging.info(
            "GPTQ: accelerate-dispatched model detected ({} devices); "
            "quantizing each layer in place.".format(len(set(map(str, device_map.values()))))
        )
    else:
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype # embedding layer의 Data type을 보고 model의 Dtype을 구한다
    # Dispatched: keep the calibration buffers on CPU (they would otherwise pin
    # nsamples*2048*hidden on cuda:0, the very GPU accelerate already filled).
    buf_device = "cpu" if dispatched else dev
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=buf_device
    )

    # FPTQuant Sn: LlamaModel.forward threads a running per-token scale layer-to-layer,
    # but GPTQ never goes through LlamaModel.forward -- it calls each decoder layer
    # directly (Catcher/hooks above). To keep GPTQ's Hessians calibrated on the SAME
    # activations the final (Sn-enabled) model will actually produce, we replicate that
    # threading here: one running scale per calibration sample, ping-ponged across layers
    # exactly like `inps`/`outs`.
    dyn_scaling = getattr(model.config, "dynamic_residual_scaling", False)
    # fp32, not `dtype` (the model's bf16/fp16): this is re-multiplied by a fresh inv_rms at
    # every block of every layer, so keeping it low-precision would compound rounding error
    # across depth -- see LlamaRMSNorm.forward's return_scale docstring.
    scales = (
        torch.ones((args.nsamples, 2048, 1), dtype=torch.float32, device=buf_device)
        if dyn_scaling
        else None
    )


    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
            # 이 과정을 통하여 모든 128개의 Sample에 1) layer에 inp  2) "attention_mask" 3) Position_ids를 가지고 온다
    layers[0] = layers[0].module

    if not dispatched:
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"] #
    position_ids = cache["position_ids"] #
    if dispatched:
        # Re-homed to each layer's own device inside the loop below.
        if attention_mask is not None:
            attention_mask = attention_mask.cpu()
        if position_ids is not None:
            position_ids = position_ids.cpu()

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]
    for i in range(len(layers)):
        if dispatched:
            # Quantize where accelerate put it; only the activations travel.
            layer = layers[i]
            layer_dev = next(layer.parameters()).device
            print(f"\nLayer {i} @ {layer_dev}:", flush=True, end=" ")
        else:
            layer_dev = torch.device(dev)
            layer = layers[i].to(layer_dev)
            print(f"\nLayer {i}:", flush=True, end=" ")
        # ReSpinQuant's T/Q/M tensors are plain attributes, not params/buffers, so they
        # do not follow the layer's device -- align them explicitly.
        _move_respin_attrs(layer, layer_dev)
        amask = attention_mask.to(layer_dev) if attention_mask is not None else None
        pids = position_ids.to(layer_dev) if position_ids is not None else None
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset: # 각 Subset마다 WeightQuantizer 설정
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                sc = scales[j].unsqueeze(0).to(layer_dev) if scales is not None else None
                outs[j] = layer(
                    inps[j].unsqueeze(0).to(layer_dev),
                    attention_mask=amask,
                    position_ids=pids,
                    residual_scale=sc,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        out_scales = torch.empty_like(scales) if scales is not None else None
        for j in range(args.nsamples):
            sc = scales[j].unsqueeze(0).to(layer_dev) if scales is not None else None
            layer_out = layer(
                inps[j].unsqueeze(0).to(layer_dev),
                attention_mask=amask,
                position_ids=pids,
                residual_scale=sc,
            )
            outs[j] = layer_out[0]
            if scales is not None:
                # layer_out[-1] is this layer's updated running scale (see
                # LlamaDecoderLayer.forward) -- this recompute pass is the one whose
                # `outs` becomes the next layer's `inps`, so it's the authoritative one.
                out_scales[j] = layer_out[-1].squeeze(0).to(buf_device)

        if not dispatched:
            layers[i] = layer.cpu()
        del layer
        del gptq
        del amask, pids
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if scales is not None:
            scales = out_scales

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


# ReSpinQuant installs the residual-correction tensors as PLAIN attributes on each
# decoder layer (layer.T_attn / .Q_attn / .M_attn / .T_ffn / .Q_ffn / .M_ffn). They are
# not nn.Parameters or registered buffers, so `layer.to(dev)` does NOT move them. They
# must be moved by hand whenever the layer changes device, or the layer forward will hit
# a cross-device error inside the residual add.
_RESPIN_ATTRS = ("T_attn", "M_attn", "Q_attn", "T_ffn", "M_ffn", "Q_ffn")


def _move_respin_attrs(module, dev):
    for name in _RESPIN_ATTRS:
        t = getattr(module, name, None)
        if torch.is_tensor(t):
            setattr(module, name, t.to(dev))


def _resolve_devices(devices):
    """Normalize a devices spec into a list of torch.device('cuda:i')."""
    if devices is None:
        devices = list(range(torch.cuda.device_count()))
    out = []
    for d in devices:
        out.append(d if isinstance(d, torch.device) else torch.device(f"cuda:{d}"))
    assert len(out) > 0, "No CUDA devices available for distributed GPTQ."
    return out


def _least_occupied(devices):
    """Pick the device with the most free VRAM right now (adapts to other processes)."""
    if len(devices) == 1:
        return devices[0]
    best, best_free = devices[0], -1
    for d in devices:
        free, _ = torch.cuda.mem_get_info(d)
        if free > best_free:
            best, best_free = d, free
    return best


@torch.no_grad()
def gptq_fwrd_distribute(model, dataloader, args, devices=None):
    """Memory-frugal GPTQ for models too large to co-reside with the calibration
    buffers on one GPU (e.g. Llama-2/3 70B).

    GPTQ is inherently SEQUENTIAL across layers -- layer i+1's calibration inputs are
    layer i's already-quantized outputs -- so layers cannot be spread across GPUs and
    run in parallel. What we distribute here is MEMORY, not compute:

      * the whole model stays on CPU;
      * the (large) inps/outs calibration buffers stay on CPU;
      * exactly ONE decoder layer at a time is streamed onto a GPU -- the least-occupied
        of `devices` -- quantized there, then moved back to CPU.

    Peak VRAM per step = one layer's weights + that layer's GPTQ Hessian (the down_proj
    Hessian dominates: intermediate^2 * 4 bytes) + one calibration sample. A single 70B
    layer therefore fits comfortably on one 24 GB card, and passing several devices lets
    consecutive layers land on whichever GPU is free (useful when the box is shared).

    Unlike accelerate's device_map="auto", there are NO AlignDevicesHooks, so this does
    not fight GPTQ's per-layer `.to(dev)` (the cause of the cuda:0/cuda:1 error).

    Requires the model to be loaded on CPU (device_map=None and no model.cuda()).
    """
    logging.info("-----GPTQ Quantization (CPU-offload / multi-GPU)-----")
    devices = _resolve_devices(devices)
    stage = devices[0]  # device used for the light-weight input-capture forward

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # --- input capture: only the pre-layer modules + layer 0 need to be on `stage`.
    #     LlamaModel.forward computes model-level rotary embeddings before the layer
    #     loop, so model.model.rotary_emb must sit on `stage` too. ---
    model.model.embed_tokens = model.model.embed_tokens.to(stage)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(stage)
    layers[0] = layers[0].to(stage)
    _move_respin_attrs(layers[0], stage)

    dtype = next(iter(model.parameters())).dtype
    nsamples = args.nsamples
    seqlen = 2048
    hidden = model.config.hidden_size
    # Calibration buffers live on CPU (4 GB each at 70B; the box has plenty of RAM).
    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device="cpu")
    # FPTQuant Sn: see the identical buffer in gptq_fwrd for why this exists -- GPTQ
    # bypasses LlamaModel.forward's own residual_scale threading, so we replicate it here.
    # fp32 (not `dtype`), same reasoning as gptq_fwrd: avoids compounding rounding error
    # across every layer this running scale is re-multiplied through.
    dyn_scaling = getattr(model.config, "dynamic_residual_scaling", False)
    scales = torch.ones((nsamples, seqlen, 1), dtype=torch.float32, device="cpu") if dyn_scaling else None
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.squeeze(0).to("cpu")
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(stage))
        except ValueError:
            pass
    layers[0] = layers[0].module

    # Everything back to CPU; keep only per-layer working set on GPU from here on.
    layers[0] = layers[0].cpu()
    _move_respin_attrs(layers[0], "cpu")
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)  # CPU
    # Keep the captured masks on CPU; they are re-homed to each layer's device per step.
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    if attention_mask is not None:
        attention_mask = attention_mask.cpu()
    if position_ids is not None:
        position_ids = position_ids.cpu()

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    for i in range(len(layers)):
        dev = _least_occupied(devices)
        print(f"\nLayer {i} -> {dev}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        _move_respin_attrs(layer, dev)
        amask = attention_mask.to(dev) if attention_mask is not None else None
        pids = position_ids.to(dev) if position_ids is not None else None

        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                if "lm_head" in name:
                    continue
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=not (args.w_asym),
                    mse=args.w_clip,
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                return tmp

            handles = [subset[n].register_forward_hook(add_batch(n)) for n in subset]
            # Stream calibration samples one at a time: only a single [1, seq, hidden]
            # activation is on the GPU at any moment.
            for j in range(nsamples):
                sc = scales[j].unsqueeze(0).to(dev) if scales is not None else None
                layer(
                    inps[j].unsqueeze(0).to(dev),
                    attention_mask=amask,
                    position_ids=pids,
                    residual_scale=sc,
                )
            for h in handles:
                h.remove()

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                    export_to_et=args.export_to_et,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        # Recompute this layer's outputs with the now-quantized weights -> store on CPU.
        out_scales = torch.empty_like(scales) if scales is not None else None
        for j in range(nsamples):
            sc = scales[j].unsqueeze(0).to(dev) if scales is not None else None
            layer_out = layer(
                inps[j].unsqueeze(0).to(dev),
                attention_mask=amask,
                position_ids=pids,
                residual_scale=sc,
            )
            outs[j] = layer_out[0].squeeze(0).to("cpu")
            if scales is not None:
                out_scales[j] = layer_out[-1].squeeze(0).to("cpu")
        if scales is not None:
            scales = out_scales

        layers[i] = layer.cpu()
        _move_respin_attrs(layers[i], "cpu")
        del layer, gptq, amask, pids
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)

        inps, outs = outs, inps  # this layer's outputs are the next layer's inputs

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args, custom_layers=None):
    """
    From GPTQ repo
    """
    # assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    if custom_layers:
        layers = custom_layers
    else:
        layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        original_device = next(layers[i].parameters()).device
        layer = layers[i].to(original_device)

        subset = quant_utils.find_qlayers( # Step 1: 각 decoderLayer에서 nn.Linear하고 nn.Embedding의 {name: module}
            layer, layers=[torch.nn.Linear, torch.nn.Embedding]
        )

        for name in subset:
            layer_weight_bits = args.w_bits
            w_groupsize = args.w_groupsize
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8
            if args.export_to_et:
                layer_weight_bits = 8  # all per channel 8 bits for executorch export
                w_groupsize = -1
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                percolumn=args.per_column,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
                weight_groupsize=w_groupsize,
                mxfp4=getattr(args, "mxfp4", False) and layer_weight_bits < 16,
                mx_block=getattr(args, "mx_block", 32),
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            q, int_weight, scale = quantizer.fake_quantize(W)
            subset[name].weight.data = q.to(next(iter(layer.parameters())).dtype) # 실제 Weight tensor의 Precision은 유지한 채로 Quantized 된 상황에 Weight를 유지한다 
            if args.export_to_et:
                subset[name].register_buffer("int_weight", int_weight)
                subset[name].register_buffer("scale", scale)
            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
        layers[i] = layer.to(device=original_device)
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers
