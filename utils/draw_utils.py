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


@torch.no_grad()
def draw_residual_norms(model, save_path, args, testloader):
    """Dejavu Fig.5(c)(d) style plot of the residual stream around each block.

    For every decoder layer we capture, on a batch of calibration tokens, the input X
    and the block output F(X) of the two residual connections  X' = X + F(X):

        * Attention block:  X = input to input_layernorm (the residual entering the
                                block), F(X) = self_attn output.
        * MLP block:        X = input to post_attention_layernorm (residual after the
                                attention add), F(X) = mlp output.

    We take the per-token l2 norm (over the hidden dim), then plot the median across
    tokens per layer with a shaded inter-quartile band -- ||X|| is much larger than
    ||F(X)||, which is why the embedding changes slowly across depth.

    Note: with ReSpinQuant the residual actually added is a basis-corrected X; here we
    report the raw residual norm, which is the quantity Dejavu measures.
    """
    os.makedirs(save_path, exist_ok=True)
    dev = model.device

    # --- build a [B, seqlen] batch of random calibration windows ---
    n_samples = int(getattr(args, "residual_norm_samples", 4))
    seqlen = model.seqlen
    total = testloader.input_ids.shape[1]
    n_samples = max(1, min(n_samples, max(1, (total - 1) // seqlen)))
    windows = []
    for _ in range(n_samples):
        s = random.randint(0, total - seqlen - 1)
        windows.append(testloader.input_ids[:, s:s + seqlen])
    inp = torch.cat(windows, dim=0).to(dev)  # [B, seqlen]

    layers = model.model.layers

    # --- capture layer-0 input + forward kwargs via a Catcher ---
    caught = {"inp": None, "attention_mask": None, "position_ids": None,
              "position_embeddings": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, x, **kwargs):
            caught["inp"] = x
            caught["attention_mask"] = kwargs.get("attention_mask")
            caught["position_ids"] = kwargs.get("position_ids")
            caught["position_embeddings"] = kwargs.get("position_embeddings")
            raise ValueError

    layers[0] = Catcher(layers[0])
    if hasattr(layers[0].module, "attention_type"):
        layers[0].attention_type = layers[0].module.attention_type
    try:
        model(inp)
    except ValueError:
        pass
    layers[0] = layers[0].module
    attention_mask = caught["attention_mask"]
    position_ids = caught["position_ids"]
    position_embeddings = caught["position_embeddings"]
    torch.cuda.empty_cache()

    def token_norms(t):
        # t: [B, S, H] -> per-token l2 norm -> flat float32 CPU vector of length B*S
        if isinstance(t, tuple):
            t = t[0]
        return t.detach().float().norm(dim=-1).reshape(-1).cpu()

    # per-layer aggregates: (median, q25, q75)
    stats = {k: [] for k in ("X_attn", "F_attn", "X_mlp", "F_mlp")}

    cur_inp = caught["inp"]
    for i in tqdm(range(len(layers)), desc="(Residual norms) Layers"):
        layer = layers[i]
        layer_dev = next(layer.parameters()).device
        rec = {}

        def cap(key):
            def hook(m, inputs, output):
                if key.startswith("X") and not args.dynamic_residual_scaling:          # residual = input to the layernorm
                    rec[key] = token_norms(inputs[0])
                else:                            # F(X) = block (attn/mlp) output
                    rec[key] = token_norms(output)
            return hook

        handles = [
            layer.input_layernorm.register_forward_hook(cap("X_attn")),
            layer.self_attn.register_forward_hook(cap("F_attn")),
            layer.post_attention_layernorm.register_forward_hook(cap("X_mlp")),
            layer.mlp.register_forward_hook(cap("F_mlp")),
        ]
        out = layer(cur_inp.to(layer_dev), attention_mask=attention_mask,
                    position_ids=position_ids, position_embeddings=position_embeddings)
        for h in handles:
            h.remove()

        for key in stats:
            v = rec[key].numpy()
            stats[key].append((np.median(v), np.percentile(v, 25), np.percentile(v, 75)))

        cur_inp = out[0] if isinstance(out, tuple) else out

    # --- plot: two panels (attention / MLP), matching Dejavu Fig.5(c)(d) ---
    stats = {k: np.array(v) for k, v in stats.items()}  # each [L, 3]
    x = np.arange(len(layers))

    def panel(ax, Xk, Fk, title):
        for key, color, label in ((Xk, "tab:orange", "||X||"),
                                   (Fk, "tab:purple", "||F(X)||")):
            med, lo, hi = stats[key][:, 0], stats[key][:, 1], stats[key][:, 2]
            ax.plot(x, med, color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.25)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Norm")
        ax.set_title(title)
        ax.legend()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panel(axes[0], "X_attn", "F_attn", "Residual Around Attention")
    panel(axes[1], "X_mlp", "F_mlp", "Residual Around MLP")
    fig.tight_layout()
    out_png = os.path.join(save_path, "residual_norms.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # also dump the raw numbers so the plot can be reproduced / compared across runs
    np.savez(os.path.join(save_path, "residual_norms.npz"),
             layer=x, **{k: stats[k] for k in stats})
    logging.info("Residual-norm figure saved to {}".format(out_png))
    return out_png


@torch.no_grad()
def draw_norm_distributions(model, save_path, args):
    """Plot the weight (gain) distribution of every RMSNorm in the model.

    Collects all RMSNorm modules -- each decoder layer's input_layernorm and
    post_attention_layernorm plus the final model.norm (Llama or Qwen) -- and writes,
    under `save_path`:

      * per_norm/<qualified_name>.png : histogram of each RMSNorm's weight values
      * rmsnorm_weight_distribution.png : median + 1/99% band of the weights vs layer,
        split by role (input_layernorm / post_attention_layernorm) with the final norm
        as a reference line
      * rmsnorm_weights.npz : the raw weight vectors, keyed by module name

    Caveat: rotation (R1) fuses LayerNorm gains into the adjacent linear weights and
    resets RMSNorm weights to 1. Run WITHOUT --rotate (or before fusion) to see the
    original learned gain distribution; a warning is logged if the weights are all ~1.
    """
    import re
    os.makedirs(save_path, exist_ok=True)

    norms = []
    for name, m in model.named_modules():
        if type(m).__name__.endswith("RMSNorm") and getattr(m, "weight", None) is not None:
            norms.append((name, m.weight.detach().float().cpu().numpy().reshape(-1)))
    if not norms:
        logging.warning("No RMSNorm modules found; nothing to draw.")
        return None

    pooled = np.concatenate([w for _, w in norms])
    if np.allclose(pooled, 1.0, atol=1e-3):
        logging.warning(
            "All RMSNorm weights are ~1.0 -- LayerNorm was likely fused into adjacent "
            "linears (rotation path). Run without --rotate to see the original gains."
        )

    # --- per-norm histograms ---
    hist_dir = os.path.join(save_path, "per_norm")
    os.makedirs(hist_dir, exist_ok=True)
    for name, w in norms:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(w, bins=100, color="tab:blue", alpha=0.8)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("RMSNorm weight value")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(hist_dir, name.replace(".", "_") + ".png"), dpi=120)
        plt.close(fig)

    # --- summary: distribution vs layer, grouped by role ---
    def layer_idx(name):
        mo = re.search(r"layers\.(\d+)\.", name)
        return int(mo.group(1)) if mo else None

    groups = {"input_layernorm": {}, "post_attention_layernorm": {}}
    final_norm = None
    for name, w in norms:
        li = layer_idx(name)
        if li is not None and "input_layernorm" in name:
            groups["input_layernorm"][li] = w
        elif li is not None and "post_attention_layernorm" in name:
            groups["post_attention_layernorm"][li] = w
        elif li is None:
            final_norm = w  # model.norm

    def series(d):
        xs = sorted(d.keys())
        med = np.array([np.median(d[i]) for i in xs])
        p1 = np.array([np.percentile(d[i], 1) for i in xs])
        p99 = np.array([np.percentile(d[i], 99) for i in xs])
        return np.array(xs), med, p1, p99

    fig, ax = plt.subplots(figsize=(8, 4))
    for role, color in (("input_layernorm", "tab:blue"),
                        ("post_attention_layernorm", "tab:red")):
        if groups[role]:
            xs, med, p1, p99 = series(groups[role])
            ax.plot(xs, med, color=color, label=role + " (median)")
            ax.fill_between(xs, p1, p99, color=color, alpha=0.2, label=role + " (1/99%)")
    if final_norm is not None:
        ax.axhline(np.median(final_norm), color="k", ls="--", lw=1,
                   label="final norm (median)")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("RMSNorm weight")
    ax.set_title("RMSNorm weight distribution across layers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    summ = os.path.join(save_path, "rmsnorm_weight_distribution.png")
    fig.savefig(summ, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(os.path.join(save_path, "rmsnorm_weights.npz"),
             **{name.replace(".", "_"): w for name, w in norms})
    logging.info("RMSNorm distribution ({} norms) saved to {}".format(len(norms), save_path))
    return summ


def _kurtosis_lastdim(t):
    """Mean over tokens of the per-token kurtosis along the hidden dim.

    t : [..., H]. Kurtosis is computed over the last dim for every row (Pearson,
    normal ~ 3), then arithmetic-averaged over all rows -> a single scalar.
    """
    t = t.detach().float()
    mu = t.mean(dim=-1, keepdim=True)
    diff = t - mu
    var = (diff ** 2).mean(dim=-1)
    m4 = (diff ** 4).mean(dim=-1)
    kurt = m4 / (var ** 2 + 1e-12)          # [...,] one kurtosis per token
    return kurt.mean().item()               # arithmetic average over all tokens


@torch.no_grad()
def draw_norm_prepost(model, save_path, args, testloader):
    """Compare the activation statistics right BEFORE vs right AFTER each RMSNorm.

    A single calibration forward is run with a hook on every RMSNorm module. Each hook
    sees the input (X, before the norm) and the output (after the norm). We produce, under
    `save_path`:

      * Kurtosis/  : a Dejavu-style x-y plot (x = Transformer layer) of the mean per-token
                     kurtosis before vs after the norm, one panel for input_layernorm and
                     one for post_attention_layernorm, plus kurtosis.npz.
      * Box Plot/  : QuaRot-style box plots of the before/after activations
                     (layers_<i>/<norm>/{before,after}/box_plot.png).
      * 3d plot/   : 3-D magnitude surfaces of the before/after activations
                     (layers_<i>/<norm>/{before,after}/3d_plot.png).

    Kurtosis is computed over the hidden dim (dim=-1) per token, then averaged (see
    _kurtosis_lastdim). Box/3-D plots are heavy; use --layer_limit to cap which layers
    get them (kurtosis is always computed for all layers).
    """
    import re
    os.makedirs(save_path, exist_ok=True)
    kurt_dir = os.path.join(save_path, "Kurtosis")
    box_dir = os.path.join(save_path, "Box Plot")
    d3_dir = os.path.join(save_path, "3d plot")
    for d in (kurt_dir, box_dir, d3_dir):
        os.makedirs(d, exist_ok=True)

    dev = model.device
    seqlen = model.seqlen
    total = testloader.input_ids.shape[1]
    s = random.randint(0, total - seqlen - 1)
    inp = testloader.input_ids[:, s:s + seqlen].to(dev)

    n_layers = len(model.model.layers)
    layer_limit = getattr(args, "layer_limit", -1)

    def should_plot(li):
        if layer_limit is None or layer_limit == -1:
            return True
        return li is not None and li < layer_limit

    def layer_idx(name):
        mo = re.search(r"layers\.(\d+)\.", name)
        return int(mo.group(1)) if mo else None

    # name -> {"before": kurt, "after": kurt}
    kurt = {}

    def make_hook(name):
        li = layer_idx(name)

        def hook(m, inputs, output):
            x_in = inputs[0]
            x_out = output[0] if isinstance(output, tuple) else output
            kurt[name] = {"before": _kurtosis_lastdim(x_in),
                          "after": _kurtosis_lastdim(x_out)}
            # heavy per-tensor plots, only for the requested layers
            if should_plot(li):
                tag = "final" if li is None else f"layers_{li}"
                short = name.split(".")[-1]
                for phase, ten in (("before", x_in), ("after", x_out)):
                    t = ten.detach().float().cpu()
                    if t.dim() == 2:
                        t = t.unsqueeze(0)          # [S,H] -> [1,S,H]
                    leaf_box = os.path.join(box_dir, tag, short, phase)
                    leaf_3d = os.path.join(d3_dir, tag, short, phase)
                    os.makedirs(leaf_box, exist_ok=True)
                    os.makedirs(leaf_3d, exist_ok=True)
                    plot_boxplot(t, leaf_box, li if li is not None else n_layers,
                                 f"{short}.{phase}", False)
                    plot_3d_map(t, leaf_3d, li if li is not None else n_layers,
                                f"{short}.{phase}", False)
        return hook

    handles = []
    for name, mod in model.named_modules():
        if type(mod).__name__.endswith("RMSNorm") and getattr(mod, "weight", None) is not None:
            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        model(inp)
    finally:
        for h in handles:
            h.remove()
    torch.cuda.empty_cache()

    # --- Kurtosis x-y plot: before vs after, per role, across layers ---
    groups = {"input_layernorm": {}, "post_attention_layernorm": {}}
    final = None
    for name, kv in kurt.items():
        li = layer_idx(name)
        if li is not None and "input_layernorm" in name:
            groups["input_layernorm"][li] = kv
        elif li is not None and "post_attention_layernorm" in name:
            groups["post_attention_layernorm"][li] = kv
        elif li is None:
            final = kv

    def panel(ax, d, title):
        xs = sorted(d.keys())
        before = [d[i]["before"] for i in xs]
        after = [d[i]["after"] for i in xs]
        ax.plot(xs, before, color="tab:orange", label="before RMSNorm")
        ax.plot(xs, after, color="tab:purple", label="after RMSNorm")
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Kurtosis (mean over tokens)")
        ax.set_title(title)
        ax.legend()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panel(axes[0], groups["input_layernorm"], "Around input_layernorm")
    panel(axes[1], groups["post_attention_layernorm"], "Around post_attention_layernorm")
    fig.tight_layout()
    kpng = os.path.join(kurt_dir, "kurtosis_prepost.png")
    fig.savefig(kpng, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # dump raw kurtosis numbers
    def arr(d, key):
        xs = sorted(d.keys())
        return np.array([d[i][key] for i in xs])
    np.savez(
        os.path.join(kurt_dir, "kurtosis.npz"),
        in_layer=np.array(sorted(groups["input_layernorm"].keys())),
        in_before=arr(groups["input_layernorm"], "before"),
        in_after=arr(groups["input_layernorm"], "after"),
        post_layer=np.array(sorted(groups["post_attention_layernorm"].keys())),
        post_before=arr(groups["post_attention_layernorm"], "before"),
        post_after=arr(groups["post_attention_layernorm"], "after"),
        final_before=np.array([final["before"]]) if final else np.array([]),
        final_after=np.array([final["after"]]) if final else np.array([]),
    )
    logging.info("Pre/Post-RMSNorm kurtosis + box/3d plots saved to {}".format(save_path))
    return kpng
