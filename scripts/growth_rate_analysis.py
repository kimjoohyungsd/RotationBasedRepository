# coding=utf-8
"""Per-DecoderLayer peak group |x|_max growth rate, at matmul boundaries
({Wq,Wk,Wv}, {Wo}, {Wup,Wgate}, {Wdown}), across three rotation treatments:

  1) no_rotation      -- raw model, untouched
  2) global_rotation   -- fuse_layer_norms + rotation_utils.rotate_model
                          (online_r2=True for Wo, diagonal=False -> full
                          hidden_size-wide R1) + apply_r3_r4.rotate_model
                          (diagonal=False -> the full R4 Hadamard on W_down)
  3) block_rotation     -- same pipeline, diagonal=True, diagonal_size=32
                          (matches the MXFP4 quantization group exactly --
                          R1/R2/R4 all become block-diagonal Hadamards
                          reused per group instead of one global rotation)

Only q_proj's input ("Attention Input") and up_proj's input ("MLP Input")
are captured/plotted (k_proj/v_proj share q_proj's input; gate_proj shares
up_proj's) -- Wo and W_down still get the online_r2 / full-R4 treatment
above because a wrong rotation there would corrupt every later layer's
residual stream, not because their own inputs are plotted.

For each layer L and input type, "peak" = the single largest per-32-group
abs-max over the whole captured activation (real wikitext2 text, 2048
tokens, one token per row, exactly matching the actual MXFP4 quantization
grouping). Growth rate = peak(condition, L) / peak(no_rotation, L).

    python scripts/growth_rate_analysis.py
    python scripts/growth_rate_analysis.py --models Llama-2-7b-hf

Figures saved to <repo>/figures/growth_rate/<model>/.
"""
import argparse
import json
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from utils import fuse_norm_utils, data_utils  # noqa: E402
from eval_utils import rotation_utils  # noqa: E402
from train_utils import apply_r3_r4  # noqa: E402

GROUP_SIZE = 32
# A shorter real (still text-filled, non-padded) window than the usual PPL-eval
# 2048: at 2048 tokens the per-block max is a max over so many different
# tokens' distributions that layer-to-layer / block-to-block growth patterns
# wash out. --seqlen overrides this.
SEQLEN = 50

MODELS = {
    "Llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "llama"),
    "Llama-3.1-8B": ("meta-llama/Llama-3.1-8B", "llama"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "qwen3"),
}
CONDITIONS = ["no_rotation", "global_rotation", "block_rotation"]
COND_LABEL = {
    "no_rotation": "No Rotation",
    "global_rotation": "Global Rotation",
    "block_rotation": f"Block Rotation (diag={GROUP_SIZE})",
}


def make_rot_args(diagonal: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        diagonal=diagonal,
        diagonal_size=GROUP_SIZE,
        offload=False,
        online_r2=True,       # Wo: online (exact-Hadamard) per-head rotation
        optimized_rotation_path=None,
        rotate_mode="hadamard",
        deactivate_r1=False,
        deactivate_r2=False,
        permute=False,        # so apply_r3_r4 takes the `diagonal` branch, not `permute`
        target_layer_indices=None,
    )


def load_model(hf_id: str, arch: str, seqlen: int = SEQLEN):
    if arch == "qwen3":
        from eval_utils.modeling_qwen3 import Qwen3ForCausalLM as ModelClass
    else:
        from eval_utils.modeling_llama import LlamaForCausalLM as ModelClass
    config = AutoConfig.from_pretrained(hf_id)
    model = ModelClass.from_pretrained(hf_id, config=config, torch_dtype=torch.float16,
                                       low_cpu_mem_usage=True)
    # eval_utils.main.ptq_model normally sets this (FPTQuant "Sn" toggle);
    # since we skip ptq_model here, set it directly -- their forward()
    # dereferences it unconditionally (no getattr default) further down the
    # decoder stack.
    model.config.dynamic_residual_scaling = False
    model.seqlen = seqlen
    return model


def apply_condition(model, condition: str):
    # fuse_layer_norms is a LOSSLESS reparameterization (it moves each LN's
    # per-channel gamma into the following Linear's weight and sets the LN's
    # own weight to all-ones -- the function the network computes is
    # unchanged). But it changes what a forward hook on q_proj/up_proj
    # actually SEES: pre-fusion the hook sees norm_output * gamma; post-fusion
    # it sees the raw norm_output, with gamma already folded into the weight
    # matrix instead. If "no_rotation" skips fusion while the rotated
    # conditions require it (rotate_model/apply_r3_r4 assume fused weights),
    # every block's captured value shifts by that per-channel gamma alone --
    # e.g. Llama-2-7b-hf layer 10's post_attention_layernorm gamma is <1.0 on
    # every single channel (mean 0.237), which alone manufactures a 100%
    # "growth rate" with zero rotation involved. Fusing here (for every
    # condition -- see "no_rotation_unfused" below for the deliberate
    # exception) means all three primary conditions capture activations on
    # the same basis, so any growth/reduction shown is actually attributable
    # to the rotation.
    if condition == "no_rotation_unfused":
        # Deliberately skips fuse_layer_norms -- the pre-fix baseline, kept
        # as an explicit, opt-in comparison point (see
        # figures/growth_rate/<model>/layer_NN/unfused_baseline/) so the
        # confound above is visible side by side with the corrected version,
        # not just asserted in a comment.
        return
    fuse_norm_utils.fuse_layer_norms(model)
    if condition == "no_rotation":
        return
    args = make_rot_args(diagonal=(condition == "block_rotation"))
    rotation_utils.rotate_model(model, args, None)
    apply_r3_r4.rotate_model(model, args)


def get_real_input_ids(hf_id: str, seqlen: int = SEQLEN) -> torch.Tensor:
    """A single real, text-filled seqlen-token window off the front of the
    concatenated WikiText-2 *test* split (data_utils.get_wikitext2's own
    eval_mode=True path) -- no padding, exactly what main.py's own PPL loop
    slices for windows 0..N."""
    tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    testenc = data_utils.get_wikitext2(seqlen=seqlen, tokenizer=tokenizer, eval_mode=True)
    return testenc.input_ids[:, :seqlen]


@torch.no_grad()
def collect_block_max(model, input_ids, group_size: int = GROUP_SIZE):
    """Per layer, per input type: the per-BLOCK max abs value, kept PER TOKEN
    -- [bsz, seq, hidden] -> [bsz, seq, group_num, group_size] -> max over the
    last (within-group) dim only, shape [bsz, seq, group_num]. Nothing is
    aggregated across tokens here (that used to happen inside this hook, via
    amax(dim=(0,1,3))); callers that want the old whole-window aggregate ("the
    worst block over the whole window") or a single token's own block profile
    (Figure 9/10's actual "one real token's block values" bar chart) both
    derive it from this same per-token tensor."""
    layers = model.model.layers
    n = len(layers)
    block_max = {"attn": [None] * n, "mlp": [None] * n}

    def make_hook(key, idx):
        def hook(_m, x, _y):
            t = x[0] if isinstance(x, tuple) else x
            t = t.detach().float()                       # [bsz, seq, hidden]
            group_num = t.shape[-1] // group_size
            t = t.reshape(t.shape[0], t.shape[1], group_num, group_size)  # [bsz,seq,group_num,group_size]
            block_max[key][idx] = t.abs().amax(dim=-1).cpu()  # [bsz, seq, group_num]
        return hook

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook("attn", i)))
        hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook("mlp", i)))

    dev = next(model.parameters()).device
    model(input_ids.to(dev))

    for h in hooks:
        h.remove()
    return block_max


def block_bar_chart(ax, base_blocks, treated_blocks, base_label, treated_label, title):
    """Figure 9/10-style overlaid bar chart: X=block index, Y=max abs value,
    two semi-transparent series, with a Growth-rate/Reduction-rate badge.

    Growth/reduction rate here is a *count* over blocks, not a magnitude:
    for each block index (same position in both same-size lists), diff =
    treated - base; growth_rate = %(diff > 0), reduction_rate = %(diff < 0)
    (blocks with diff == 0 count toward neither). The two always sum to
    <=100% and are shown together -- this says what fraction of blocks moved
    which way, as opposed to how far the single most extreme block moved.
    """
    n = len(base_blocks)
    x = list(range(n))
    ax.bar(x, base_blocks, width=1.0, color="#a6cee3", alpha=0.7, label=base_label)
    ax.bar(x, treated_blocks, width=1.0, color="#1f4e79", alpha=0.55, label=treated_label)
    ax.set_xlabel("Block Index")
    ax.set_ylabel("Max Absolute Value")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    base_t = torch.tensor(base_blocks)      # [group_num]
    treated_t = torch.tensor(treated_blocks)  # [group_num]
    diff = treated_t - base_t               # same-index (per-block) comparison
    total = diff.numel()
    growth_rate = (diff > 0).sum().item() / total * 100.0
    reduction_rate = (diff < 0).sum().item() / total * 100.0

    ax.text(0.02, 0.96, f"Growth rate: {growth_rate:.1f}%", transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d62728", alpha=0.25))
    ax.text(0.02, 0.87, f"Reduction rate: {reduction_rate:.1f}%", transform=ax.transAxes,
            fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ca02c", alpha=0.25))


def save_layer_comparisons(save_dir, base_blocks_by_key, treated_blocks_by_key, base_label,
                           treated_label, condition, model_name, layer_idx, token_note=""):
    """One whole-window-aggregate chart, given already-[group_num] block lists
    per input type (the caller decides whether that's an amax over the whole
    window or one specific token's own row)."""
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, type_label in zip(axes, ("attn", "mlp"), ("Attention Input (q_proj)", "MLP Input (up_proj)")):
        block_bar_chart(ax, base_blocks_by_key[key], treated_blocks_by_key[key], base_label, treated_label,
                        f"{type_label}: Layer {layer_idx}{token_note}")
    fig.suptitle(f"{model_name} -- Layer {layer_idx}{token_note}: {base_label} vs. {treated_label}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"no_vs_{condition}.png"), dpi=150)
    plt.close(fig)


def run_model(model_name, hf_id, arch, out_root, seqlen=SEQLEN, num_token_samples=5, token_seed=0):
    print(f"\n{'=' * 78}\n{model_name}  (seqlen={seqlen})\n{'=' * 78}")
    input_ids = get_real_input_ids(hf_id, seqlen=seqlen)
    print(f"input_ids: {tuple(input_ids.shape)}")

    # "no_rotation_unfused" is an explicit, opt-in extra: the pre-fix baseline
    # (fuse_layer_norms skipped) kept around so the LN-fusion confound (see
    # apply_condition) is visible side by side with the corrected comparison,
    # under layer_NN/unfused_baseline/, instead of just asserted in a comment.
    all_conditions = CONDITIONS + ["no_rotation_unfused"]
    all_blocks = {}   # condition -> {"attn"|"mlp": [tensor[bsz, seq, group_num] per layer]}
    for condition in all_conditions:
        print(f"-- {condition} --")
        model = load_model(hf_id, arch, seqlen=seqlen)
        apply_condition(model, condition)
        model = model.cuda().eval()
        model.config.use_cache = False
        all_blocks[condition] = collect_block_max(model, input_ids)
        del model
        torch.cuda.empty_cache()

    out_dir = os.path.join(out_root, model_name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(all_blocks, os.path.join(out_dir, "block_max.pt"))

    n_layers = len(all_blocks["no_rotation"]["attn"])
    layers_x = list(range(n_layers))

    # per-layer peak (= max over blocks AND over every token in the window),
    # used by the summary plots below and written out as the same compact
    # peaks.json as before -- amax is associative, so maxing over the
    # within-group dim first (in the hook) and (bsz, seq) here is identical
    # to the old single amax(dim=(0,1,3)).
    all_peaks = {
        condition: {
            key: [t.amax(dim=(0, 1)).max().item() for t in all_blocks[condition][key]]
            for key in ("attn", "mlp")
        }
        for condition in all_conditions
    }
    with open(os.path.join(out_dir, "peaks.json"), "w") as f:
        json.dump(all_peaks, f, indent=2)

    # A handful of real token positions, fixed by seed so the same tokens are
    # used across every layer/model/condition (an apples-to-apples snapshot
    # of ONE real token's own block profile, per Figure 9/10's actual data --
    # a max over many tokens smooths that out).
    import random
    token_indices = sorted(random.Random(token_seed).sample(range(seqlen), min(num_token_samples, seqlen)))
    print(f"sampled token indices (seed={token_seed}): {token_indices}")

    # Per-decoder-layer block-index bar charts (Figure 9/10 style), split by
    # input type, comparing No Rotation against EACH rotation treatment --
    # both the whole-window aggregate (as before) and one chart per sampled
    # token position, under layer_NN/[unfused_baseline/][token_<idx>/].
    baselines = [("no_rotation", "No Rotation", "")]  # (condition key, label, subdir under layer_NN/)
    unfused_sub = "unfused_baseline"
    baselines.append(("no_rotation_unfused", "No Rotation (LN unfused)", unfused_sub))

    for layer_idx in range(n_layers):
        layer_dir = os.path.join(out_dir, f"layer_{layer_idx:02d}")
        for base_condition, base_label, subdir in baselines:
            base_root = os.path.join(layer_dir, subdir) if subdir else layer_dir
            for condition in ("global_rotation", "block_rotation"):
                base_whole = {key: all_blocks[base_condition][key][layer_idx].amax(dim=(0, 1)).tolist()
                              for key in ("attn", "mlp")}
                treated_whole = {key: all_blocks[condition][key][layer_idx].amax(dim=(0, 1)).tolist()
                                 for key in ("attn", "mlp")}
                save_layer_comparisons(base_root, base_whole, treated_whole, base_label,
                                      COND_LABEL[condition], condition, model_name, layer_idx,
                                      token_note=" (whole window)")

                for token_idx in token_indices:
                    token_dir = os.path.join(base_root, f"token_{token_idx:03d}")
                    base_tok = {key: all_blocks[base_condition][key][layer_idx][0, token_idx, :].tolist()
                               for key in ("attn", "mlp")}
                    treated_tok = {key: all_blocks[condition][key][layer_idx][0, token_idx, :].tolist()
                                  for key in ("attn", "mlp")}
                    save_layer_comparisons(token_dir, base_tok, treated_tok, base_label,
                                          COND_LABEL[condition], condition, model_name, layer_idx,
                                          token_note=f", token {token_idx}")
    print(f"saved {n_layers} layers x 2 baselines x (1 whole-window + {len(token_indices)} per-token) x 2 "
         f"comparisons of block-index bar charts under {out_dir}/layer_*/[{unfused_sub}/][token_*/]")

    # Figure 1: absolute peak group |x|_max per layer, all 3 conditions, one
    # subplot per input type.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes, ("attn", "mlp"), ("Attention Input (q_proj)", "MLP Input (up_proj)")):
        for condition in CONDITIONS:
            ax.plot(layers_x, all_peaks[condition][key], marker='o', markersize=3,
                   label=COND_LABEL[condition])
        ax.set_title(title)
        ax.set_xlabel("Decoder Layer Index")
        ax.set_ylabel("Peak group |x|_max")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_name}: peak MXFP4-group (size={GROUP_SIZE}) abs-max per layer")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "absolute_peaks.png"), dpi=150)
    plt.close(fig)

    # Figure 2/3: growth rate vs no_rotation, one figure per rotation condition,
    # X=layer, Y=growth rate, one line per input type.
    for condition in ("global_rotation", "block_rotation"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in (("attn", "Attention Input"), ("mlp", "MLP Input")):
            base = all_peaks["no_rotation"][key]
            treated = all_peaks[condition][key]
            growth = [(t / b - 1.0) * 100.0 for t, b in zip(treated, base)]
            ax.plot(layers_x, growth, marker='o', markersize=3, label=label)
        ax.axhline(0.0, color='gray', linewidth=1, linestyle='--')
        ax.set_xlabel("Decoder Layer Index")
        ax.set_ylabel("Peak group |x|_max growth rate (%) vs. No Rotation")
        ax.set_title(f"{model_name}: {COND_LABEL[condition]} vs. No Rotation")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{condition}_growth_rate.png"), dpi=150)
        plt.close(fig)

    print(f"saved figures to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument('--out_root', default=os.path.join(HERE, 'figures', 'growth_rate'))
    ap.add_argument('--seqlen', type=int, default=SEQLEN,
                    help="real, text-filled token window length pulled from the front "
                         "of WikiText-2's test split")
    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    for model_name in args.models:
        hf_id, arch = MODELS[model_name]
        run_model(model_name, hf_id, arch, args.out_root, seqlen=args.seqlen)


if __name__ == '__main__':
    main()
