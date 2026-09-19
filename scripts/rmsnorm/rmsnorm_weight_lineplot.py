# coding=utf-8
"""Line plot of RMSNorm weight (gamma) distribution vs. decoder-layer index, for
the two residual-path norms that fuse_layer_norms actually folds into the
neighbouring linears:

    input_layernorm            (feeds q/k/v_proj)
    post_attention_layernorm   (feeds gate/up_proj)

For every layer index the per-channel gamma values are reduced to a single
representative statistic (mean by default) drawn as the line, with a mean +/- std
band (fill-between, or error bars with --errorbar) showing the spread. Both norms
are drawn in one axes in distinct colours.

Reuses scripts/rmsnorm_weight_histograms.py's model loading + weight collection
(collect_rmsnorm_weights already drops Qwen3's per-head q_norm/k_norm, which are
not on the residual stream and never fused).

    python scripts/rmsnorm_weight_lineplot.py
    python scripts/rmsnorm_weight_lineplot.py --models Llama-2-7b-hf Qwen3-8B
    python scripts/rmsnorm_weight_lineplot.py --stat median --errorbar

One <model>/layerwise_lineplot.png (+ layerwise_stats.json) per model, written
under figures/rmsnorm_histograms/ alongside the histogram outputs.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from scripts.rmsnorm_weight_histograms import (  # noqa: E402
    MODELS, load_model, collect_rmsnorm_weights,
)

# Residual-path norms fuse_layer_norms folds away, in draw order, with a fixed
# colour each so the two series are always distinguishable.
TARGET_MODULES = {
    "input_layernorm": "tab:blue",
    "post_attention_layernorm": "tab:orange",
}


def layerwise_series(per_layer, module_name, stat):
    """-> (x, center, std) numpy arrays over the layers that have `module_name`.
    center is mean or median per that layer's gamma channels; std is always the
    per-layer standard deviation (the band half-width)."""
    idxs = sorted(i for i in per_layer if module_name in per_layer[i])
    x, center, std = [], [], []
    for i in idxs:
        w = per_layer[i][module_name]
        x.append(i)
        center.append(np.median(w) if stat == "median" else w.mean())
        std.append(w.std())
    return np.array(x), np.array(center), np.array(std)


def run_model(model_name, hf_id, arch, out_root, stat, errorbar):
    print(f"\n{'=' * 78}\n{model_name}\n{'=' * 78}")
    model = load_model(hf_id, arch)
    per_layer, _ = collect_rmsnorm_weights(model)
    del model

    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    summary = {"stat": stat, "modules": {}}

    for module_name, color in TARGET_MODULES.items():
        x, center, std = layerwise_series(per_layer, module_name, stat)
        if x.size == 0:
            print(f"  [skip] {module_name}: not present in {model_name}")
            continue

        ax.plot(x, center, color=color, marker="o", markersize=3, linewidth=1.6,
                label=f"{module_name} ({stat})")
        if errorbar:
            ax.errorbar(x, center, yerr=std, color=color, fmt="none",
                        elinewidth=1, capsize=2, alpha=0.7)
        else:
            ax.fill_between(x, center - std, center + std, color=color, alpha=0.20,
                            label=f"{module_name} ({stat} $\\pm$ std)")

        summary["modules"][module_name] = {
            "layer_index": x.tolist(),
            stat: center.tolist(),
            "std": std.tolist(),
        }

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5,
               label="gamma = 1.0")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("RMSNorm Weight Value")
    ax.set_title(f"{model_name}: residual-path RMSNorm weight ($\\gamma$) vs. layer index"
                 f"\n(line = per-layer {stat}, band = {stat} $\\pm$ std)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out_png = os.path.join(out_dir, "layerwise_lineplot.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    with open(os.path.join(out_dir, "layerwise_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {out_png}\nsaved {os.path.join(out_dir, 'layerwise_stats.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--out_root", default=os.path.join(HERE, "figures", "rmsnorm_histograms"))
    ap.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="per-layer representative statistic for the line (default: mean, "
                         "which pairs with the mean +/- std band).")
    ap.add_argument("--errorbar", action="store_true",
                    help="draw the +/- std spread as error bars instead of a fill-between band.")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    for model_name in args.models:
        hf_id, arch = MODELS[model_name]
        run_model(model_name, hf_id, arch, args.out_root, args.stat, args.errorbar)


if __name__ == "__main__":
    main()
