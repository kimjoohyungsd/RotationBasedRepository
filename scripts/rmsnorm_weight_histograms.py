# coding=utf-8
"""Histograms of RMSNorm weight (gamma) values, at three granularities, for
the raw (un-fused, un-rotated) model as loaded from the checkpoint:

  1) per DecoderLayer -- all norm modules *within* one layer overlaid in one
     histogram (input_layernorm, post_attention_layernorm, and for Qwen3
     also self_attn.q_norm/k_norm), so you can compare them at that depth.
  2) per module type -- one module's weights pooled across *every* layer
     (e.g. every layer's input_layernorm together) into one histogram, so
     you can see how that specific module's distribution looks model-wide.
  3) whole model -- every RMSNorm weight in the model (every layer, every
     module type, plus the final model.norm) pooled into one histogram.

Directly motivated by scripts/growth_rate_analysis.py's finding that
individual layers' gamma can sit almost entirely below 1.0 (Llama-2-7b-hf
layer 10's post_attention_layernorm: mean 0.237, max 0.264 across all 4096
channels) -- this makes that visible across the whole model instead of one
spot-checked layer.

RMSNorm modules are discovered generically (any submodule whose class name
contains "RMSNorm"), not hardcoded to input_layernorm/post_attention_layernorm,
so Qwen3's per-head q_norm/k_norm are picked up automatically.

    python scripts/rmsnorm_weight_histograms.py
    python scripts/rmsnorm_weight_histograms.py --models Llama-2-7b-hf

Figures saved to <repo>/figures/rmsnorm_histograms/<model>/.
"""
import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MODELS = {
    "Llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "llama"),
    "Llama-3.1-8B": ("meta-llama/Llama-3.1-8B", "llama"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "qwen3"),
}
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def load_model(hf_id: str, arch: str):
    if arch == "qwen3":
        from eval_utils.modeling_qwen3 import Qwen3ForCausalLM as ModelClass
    else:
        from eval_utils.modeling_llama import LlamaForCausalLM as ModelClass
    config = AutoConfig.from_pretrained(hf_id)
    return ModelClass.from_pretrained(hf_id, config=config, torch_dtype=torch.float16,
                                      low_cpu_mem_usage=True)


def collect_rmsnorm_weights(model):
    """-> (per_layer, final_norms), where
    per_layer: {layer_idx: {module_name: 1D float32 numpy array}}
    final_norms: {module_name: 1D float32 numpy array}  (not under model.layers.*)
    module_name is the submodule path with the layer index stripped, e.g.
    "input_layernorm" or "self_attn.q_norm", so the same key groups a module
    type across every layer.
    """
    per_layer, final_norms = {}, {}
    for name, module in model.named_modules():
        if "RMSNorm" not in type(module).__name__:
            continue
        w = module.weight.detach().float().cpu().numpy().reshape(-1)
        m = LAYER_RE.match(name)
        if m:
            layer_idx, module_name = int(m.group(1)), m.group(2)
            per_layer.setdefault(layer_idx, {})[module_name] = w
        else:
            final_norms[name] = w
    return per_layer, final_norms


def stats(w: np.ndarray) -> str:
    return f"n={w.size}  mean={w.mean():.3f}  std={w.std():.3f}  min={w.min():.3f}  max={w.max():.3f}"


def hist_panel(ax, series: dict, title: str, bins: int = 60):
    """series: {label: 1D array}, each drawn as a separate overlaid histogram."""
    all_vals = np.concatenate(list(series.values()))
    lo, hi = all_vals.min(), all_vals.max()
    edges = np.linspace(lo, hi, bins + 1)
    for label, w in series.items():
        ax.hist(w, bins=edges, alpha=0.55, label=f"{label}  ({stats(w)})")
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("RMSNorm weight value")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def run_model(model_name, hf_id, arch, out_root):
    print(f"\n{'=' * 78}\n{model_name}\n{'=' * 78}")
    model = load_model(hf_id, arch)
    per_layer, final_norms = collect_rmsnorm_weights(model)
    del model

    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) per DecoderLayer: all of that layer's norm modules overlaid ----
    per_layer_dir = os.path.join(out_dir, "per_layer")
    os.makedirs(per_layer_dir, exist_ok=True)
    n_layers = max(per_layer) + 1
    for layer_idx in sorted(per_layer):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        hist_panel(ax, per_layer[layer_idx], f"{model_name} -- Layer {layer_idx}: RMSNorm weights")
        fig.tight_layout()
        fig.savefig(os.path.join(per_layer_dir, f"layer_{layer_idx:02d}.png"), dpi=150)
        plt.close(fig)
    print(f"saved {len(per_layer)} per-layer histograms to {per_layer_dir}/")

    # ---- 2) per module type: pooled across every layer ----
    module_names = sorted({name for mods in per_layer.values() for name in mods})
    per_module_dir = os.path.join(out_dir, "per_module")
    os.makedirs(per_module_dir, exist_ok=True)
    module_summary = {}
    for module_name in module_names:
        pooled = np.concatenate([per_layer[i][module_name] for i in sorted(per_layer)
                                 if module_name in per_layer[i]])
        module_summary[module_name] = stats(pooled)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        hist_panel(ax, {f"all layers pooled": pooled},
                  f"{model_name} -- {module_name} (every layer pooled)")
        fig.tight_layout()
        safe_name = module_name.replace(".", "_")
        fig.savefig(os.path.join(per_module_dir, f"{safe_name}.png"), dpi=150)
        plt.close(fig)
    for name, w in final_norms.items():
        module_summary[name] = stats(w)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        hist_panel(ax, {name: w}, f"{model_name} -- {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(per_module_dir, f"{name.replace('.', '_')}.png"), dpi=150)
        plt.close(fig)
    print(f"saved {len(module_names) + len(final_norms)} per-module histograms to {per_module_dir}/")

    # ---- 3) whole model: every RMSNorm weight, pooled ----
    everything = np.concatenate(
        [w for mods in per_layer.values() for w in mods.values()] + list(final_norms.values())
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    hist_panel(ax, {"all RMSNorm weights": everything}, f"{model_name}: every RMSNorm weight, whole model")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "whole_model.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "n_layers": n_layers,
            "per_module": module_summary,
            "whole_model": stats(everything),
        }, f, indent=2)
    print(f"saved whole_model.png and summary.json to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument('--out_root', default=os.path.join(HERE, 'figures', 'rmsnorm_histograms'))
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    for model_name in args.models:
        hf_id, arch = MODELS[model_name]
        run_model(model_name, hf_id, arch, args.out_root)


if __name__ == '__main__':
    main()
