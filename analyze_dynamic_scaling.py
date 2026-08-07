# coding=utf-8
"""Visualize what FPTQuant's Sn transform (--dynamic_residual_scaling) does to the
activation distributions of a rotated model.

Sn (arXiv:2506.04985, Sec. 3.1.3) is a *pure runtime* transform: it changes no weight,
it only threads a running per-token scale through LlamaModel.forward. That means one
loaded model can produce both arms of the comparison -- we simply toggle
`model.config.dynamic_residual_scaling` between two forward passes over the *same*
tokens. Nothing else differs, so every change in the plots is attributable to Sn.

Three figures are produced:

  1. `<out>/layer<L>/<module>_3d.png`
     Token x Channel x |activation| surface, Original vs +Sn side by side.

  2. `<out>/kurtosis_sequential.png` (+ `.csv`, + `kurtosis_per_module.png`)
     x-axis: the quantizer index in exactly the order gptq_fwrd's `sequential` list walks
     them, one point per group -- (k,v,q) -> o -> (up,gate) -> down -- because every
     module inside a group consumes the same activation;
     y-axis: excess kurtosis of that activation. One line per arm.

  3. `<out>/layer<L>/down_proj_token_dist.png`
     The paper's Figure 2: per-token min / max / 1% / 99% of the down_proj input
     (the SwiGLU intermediate), Original vs +Sn.

The analysis input is one coherent wikitext-2 article tokenized from its first word, so
position 0 is a genuine [BOS] -- the token the paper's Figure 2 calls out and hides.
(Slicing a window out of the concatenated corpus, as data_utils.get_wikitext2 does for
perplexity, starts mid-sentence and contains no BOS at all.) Use --sn_article to pick a
different article or --sn_text_file to supply your own passage.

Requires --rotate: Sn folds its residual renormalization into input_layernorm /
post_attention_layernorm, which is only weightless after fuse_layer_norms (see
eval_utils/main.py::ptq_model).

Example:
    python analyze_dynamic_scaling.py \
        --input_model meta-llama/Llama-3.2-3B-Instruct \
        --rotate --rotate_mode hadamard \
        --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 \
        --sn_layers 0 1 --sn_seqlen 512 \
        --distribution_dir figures/Sn
"""

import os
from logging import Logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaTokenizerFast, PreTrainedTokenizerFast

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM
from utils import quant_utils, utils
from utils.process_args import process_args_ptq

# The capture points, in the exact order gptq_fwrd walks them. Kept as the nested
# structure (rather than a flat list) so the x-axis of figure 2 reproduces GPTQ's
# grouping, and so the group boundaries can be drawn.
SEQUENTIAL = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]
FLAT_SEQUENTIAL = [name for group in SEQUENTIAL for name in group]

# One representative per gptq group. Every module inside a `sequential` group consumes
# the *same* activation -- q/k/v all read the input_layernorm output, up/gate both read
# the post_attention_layernorm output -- which is precisely why GPTQ can quantize a group
# in one shot. Hooking all seven therefore records the identical tensor two or three
# times: same kurtosis, same 3D surface, 3x the memory. Use --sn_verify_shared to
# re-check the assumption on a given model/config.
CAPTURE_POINTS = [group[0] for group in SEQUENTIAL]
SHORT = {
    "self_attn.k_proj": "attn_in",
    "self_attn.o_proj": "o",
    "mlp.up_proj": "mlp_in",
    "mlp.down_proj": "down",
}
# What each representative actually stands for, for axis labels and figure titles.
REPRESENTS = {
    "self_attn.k_proj": "attn input (= q, k, v)",
    "self_attn.o_proj": "o_proj input",
    "mlp.up_proj": "mlp input (= up, gate)",
    "mlp.down_proj": "down_proj input (SwiGLU)",
}


# Surface resolution cap per axis. 14336 channels x 512 tokens is 7.3M quads, which
# matplotlib's 3D renderer will not finish in reasonable time.
MAX_SURFACE_CELLS = 256


def _maxpool2d(a: np.ndarray, cap: int):
    """Shrink `a` to at most `cap` cells per axis by taking the max over each block.

    Max, not stride-sampling: this figure exists to show outliers, and plain striding
    (matplotlib's rstride/cstride) would silently skip 111 of every 112 channels -- so a
    lone outlier channel, the exact thing being visualized, has a ~1% chance of surviving.
    Returns (pooled, row_factor, col_factor) so the axes can still be labelled in
    original channel/token units.
    """
    T, C = a.shape
    ry, rx = max(1, -(-T // cap)), max(1, -(-C // cap))
    if (ry, rx) == (1, 1):
        return a, 1, 1
    # Trim the ragged tail rather than padding: padding with zeros would invent
    # low-magnitude cells, padding with edge values would duplicate outliers.
    a = a[: (T // ry) * ry, : (C // rx) * rx]
    a = a.reshape(T // ry, ry, C // rx, rx).max(axis=(1, 3))
    return a, ry, rx


def excess_kurtosis(x: torch.Tensor) -> float:
    """Population excess kurtosis over every element of `x`.

    Done on-device in fp32: the down_proj input is (seq x 14336) per layer, so pulling
    it to CPU for scipy just to get one scalar would dominate the runtime. The bias
    correction scipy applies is negligible at these sample counts (>1e6).
    """
    x = x.detach().float().reshape(-1)
    d = x - x.mean()
    m2 = d.pow(2).mean()
    m4 = d.pow(4).mean()
    return float((m4 / m2.pow(2) - 3.0).item())


@torch.no_grad()
def run_pass(model, inp, enable_sn, detail_layers, capture, keep_first, log,
             verify_shared=False):
    """One forward pass over `inp`, returning per-capture-point statistics.

    Only one module per gptq `sequential` group is hooked (see CAPTURE_POINTS): the
    others consume the identical tensor, so recording them adds nothing but memory.

    Returns (kurt, detail) where
      kurt[(layer_idx, module_name)] -> float
      detail[(layer_idx, module_name)] -> fp32 CPU tensor [tokens, channels]
    `detail` is only populated for layers in `detail_layers` (the full tensors are far
    too big to keep for every layer).

    With verify_shared=True every group member is hooked as well and checked against its
    representative, so the deduplication can be re-validated whenever the model family or
    the capture point changes.
    """
    model.config.dynamic_residual_scaling = bool(enable_sn)

    kurt, detail = {}, {}
    layers = model.model.layers
    detail_layers = set(detail_layers)
    points = FLAT_SEQUENTIAL if verify_shared else CAPTURE_POINTS

    # Verification is keyed by gptq group, not by "the representative", because hooks fire
    # in call order, not in SEQUENTIAL order: LlamaAttention runs q -> k -> v while the
    # representative is k, and LlamaMLP runs gate -> up while the representative is up. So
    # compare every member against whichever member is seen *first* in that layer -- an
    # order-independent test of "all members of this group get the same tensor".
    group_of = {name: g for g, group in enumerate(SEQUENTIAL) for name in group}
    group_size = {g: len(group) for g, group in enumerate(SEQUENTIAL)}
    first_seen = {}   # (layer_idx, group_id) -> first tensor observed
    remaining = {}    # (layer_idx, group_id) -> members still to arrive
    mismatches = []

    def make_hook(layer_idx, name):
        def hook(module, args):
            x = args[0]
            if isinstance(x, tuple):
                x = x[0]
            # [batch, seq, channel] -> [seq, channel]; batch is always 1 here.
            x = x.detach().reshape(-1, x.shape[-1]).float()

            if verify_shared:
                g = group_of[name]
                key = (layer_idx, g)
                if key not in first_seen:
                    first_seen[key] = (name, x)
                    remaining[key] = group_size[g] - 1
                else:
                    ref_name, ref = first_seen[key]
                    # Exact equality, not a norm/moment fingerprint: the claim being
                    # tested is that these are the same data, and moments can agree for
                    # genuinely different tensors.
                    if not torch.equal(ref, x):
                        mismatches.append((layer_idx, name, ref_name))
                    remaining[key] -= 1
                    if remaining[key] == 0:
                        # Whole group has arrived; release the reference so peak memory
                        # stays at ~one hidden-state tensor per in-flight group.
                        del first_seen[key], remaining[key]
                if name not in CAPTURE_POINTS:
                    return  # non-representatives contribute no statistics
            # Drop [BOS] from the *statistic* too, not just the plot: its activation is
            # orders of magnitude above everything else (Sun et al., 2024), so leaving it
            # in makes kurtosis a measure of that single token and nothing else -- it
            # pushes layer 1 of Llama-3.1-8B from ~1e4 to ~7e6. The paper hides the same
            # token in Figure 2.
            if not keep_first:
                x = x[1:]
            kurt[(layer_idx, name)] = excess_kurtosis(x)
            if layer_idx in detail_layers:
                detail[(layer_idx, name)] = x.cpu()
        return hook

    handles = []
    for i, layer in enumerate(layers):
        modules = dict(layer.named_modules())
        for name in points:
            target = modules.get(name)
            if target is None:
                continue
            if capture == "linear" and isinstance(target, quant_utils.ActQuantWrapper):
                # gptq_fwrd's `sequential` names carry a trailing ".module"; tapping it
                # means observing the activation *after* the online Hadamard and the
                # activation quantizer have already reshaped it.
                target = target.module
            handles.append(target.register_forward_pre_hook(make_hook(i, name)))

    try:
        model(inp)
    finally:
        for h in handles:
            h.remove()
        first_seen.clear()
        remaining.clear()

    if verify_shared:
        if mismatches:
            raise AssertionError(
                "These modules do NOT share their gptq group's input, so capturing only "
                "the representative would lose information: "
                + ", ".join(f"layers.{i}.{n} != layers.{i}.{r}" for i, n, r in mismatches)
            )
        log.info(
            "verify_shared: every gptq group member matched its representative exactly "
            "({} comparisons)".format(len(layers) * (len(FLAT_SEQUENTIAL) - len(CAPTURE_POINTS)))
        )

    log.info(
        "{} pass done: {} capture points".format(
            "+Sn" if enable_sn else "Original", len(kurt)
        )
    )
    return kurt, detail


def plot_3d(base, sn, out_path, layer_idx, name):
    """Token x Channel x |activation| surface for both arms.

    `base`/`sn` already have [BOS] removed by run_pass, so the surface and the kurtosis
    in the titles describe the exact same elements.

    Each panel keeps its own z-range: Sn rescales tokens by the running 1/||residual||,
    so the arms sit at different absolute magnitudes and a shared axis would flatten one
    of them. What the figure is meant to show is the *shape* of the spikes, not their
    absolute height -- the kurtosis in the title carries the scale-free comparison.
    """
    fig = plt.figure(figsize=(13, 5.5))
    data = [_maxpool2d(np.abs(arr.numpy()), MAX_SURFACE_CELLS) for arr in (base, sn)]

    for k, ((a, ry, rx), title) in enumerate(zip(data, ("Original", "With $S_n$"))):
        T, C = a.shape
        X, Y = np.meshgrid(np.arange(C) * rx, np.arange(T) * ry)
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        surf = ax.plot_surface(X, Y, a, cmap="coolwarm", linewidth=0, antialiased=False)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Token")
        pool = "" if (ry, rx) == (1, 1) else f"\nmax-pooled {ry}x{rx} (token x channel)"
        ax.set_title(
            f"{title}\nlayers.{layer_idx} {REPRESENTS[name]}{pool}", fontsize=10
        )
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.1)

    plt.tight_layout()
    # tight_layout measures the 3D axes' bounding box, not the text above it, so the
    # three-line titles get clipped without an explicit top margin.
    fig.subplots_adjust(top=0.82)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_token_distribution(base, sn, kurt_base, kurt_sn, out_path, layer_idx, name):
    """Figure 2 of the paper: per-token min / max / 1% / 99% across channels."""
    # Independent y-axes on purpose. Sn rescales every token by the running 1/||residual||,
    # so the two arms live at genuinely different absolute magnitudes; sharing the axis
    # flattens the Original panel into a line at zero and hides its outlier structure.
    # The paper's Figure 2 likewise gives each panel its own range.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    panels = [
        (base, kurt_base, f"(a) Block {layer_idx} - Original"),
        (sn, kurt_sn, f"(b) Block {layer_idx} - With $S_n$"),
    ]
    for ax, (arr, k, title) in zip(axes, panels):
        a = arr.numpy()
        x = np.arange(a.shape[0])
        ax.plot(x, a.min(axis=1), color="tab:blue", lw=0.8, label="min")
        ax.plot(x, a.max(axis=1), color="tab:orange", lw=0.8, label="max")
        ax.plot(x, np.percentile(a, 1, axis=1), color="green", lw=0.8, label="q=1%")
        ax.plot(x, np.percentile(a, 99, axis=1), color="red", lw=0.8, label="q=99%")
        ax.set_xlabel("Token")
        ax.set_title(f"{title}\nKurtosis: {k:.0f}", fontsize=10)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("FFN intermediate activation")

    delta = 100.0 * (kurt_sn - kurt_base) / abs(kurt_base) if kurt_base else float("nan")
    fig.suptitle(
        f"layers.{layer_idx}.{name} input - kurtosis {kurt_base:.0f} -> {kurt_sn:.0f} "
        f"({delta:+.1f}%)", fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_kurtosis_sequential(kurt_base, kurt_sn, n_layers, out_dir):
    """x = flattened gptq `sequential` index, y = excess kurtosis, one line per arm."""
    rows, xs = [], []
    for i in range(n_layers):
        for name in CAPTURE_POINTS:
            key = (i, name)
            if key not in kurt_base:
                continue
            xs.append(len(xs))
            rows.append((len(xs) - 1, i, name, kurt_base[key], kurt_sn[key]))

    if not rows:
        return

    idx = np.array([r[0] for r in rows])
    kb = np.array([r[3] for r in rows])
    ks = np.array([r[4] for r in rows])

    csv_path = os.path.join(out_dir, "kurtosis_sequential.csv")
    with open(csv_path, "w") as f:
        f.write("seq_index,layer,module,represents,kurtosis_original,kurtosis_sn\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{REPRESENTS[r[2]]},{r[3]:.6f},{r[4]:.6f}\n")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx, kb, color="tab:red", lw=1.0, marker="o", ms=2.2, label="Original")
    ax.plot(idx, ks, color="tab:blue", lw=1.0, marker="o", ms=2.2, label="With $S_n$")
    # Excess kurtosis can dip slightly below zero for well-behaved tensors, so a plain
    # log scale would drop points; symlog keeps them and still compresses the 1e3 peaks.
    ax.set_yscale("symlog", linthresh=1.0)

    per_layer = len(CAPTURE_POINTS)
    for i in range(0, n_layers + 1):
        ax.axvline(i * per_layer - 0.5, color="0.85", lw=0.6, zorder=0)
    step = max(1, n_layers // 16)
    ticks = [i * per_layer + per_layer / 2 for i in range(0, n_layers, step)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"L{i}" for i in range(0, n_layers, step)])

    ax.set_xlabel(
        "gptq_fwrd sequential index (one point per group -- members share their input)   "
        "[ attn in (q,k,v) -> o_proj -> mlp in (up,gate) -> down_proj ] per layer"
    )
    ax.set_ylabel("Excess kurtosis")
    ax.set_title("Activation kurtosis along the GPTQ sequential order")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kurtosis_sequential.png"), dpi=200)
    plt.close(fig)

    # The single axis packs every layer together; the same data faceted by capture point
    # is what you actually read a per-location trend off of.
    # No sharex: each panel carries its own "Layer" label, so hiding the top row's tick
    # numbers would leave a labelled axis with nothing to read against.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes.flat, CAPTURE_POINTS):
        ls = [i for i in range(n_layers) if (i, name) in kurt_base]
        vb = [kurt_base[(i, name)] for i in ls]
        vs = [kurt_sn[(i, name)] for i in ls]
        ax.plot(ls, vb, color="tab:red", lw=1.2, label="Original")
        ax.plot(ls, vs, color="tab:blue", lw=1.2, label="With $S_n$")
        # Per-panel scale. The post-rotation layernorm outputs sit at |kurtosis| < 1,
        # where a symlog axis degenerates into unreadable "1.13 x 10^-2" ticks; only the
        # panels that actually span decades get the log treatment.
        if max(max(vb), max(vs)) > 10.0:
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{REPRESENTS[name]}\n[{name}]", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Layer")
    axes.flat[0].set_ylabel("Excess kurtosis")
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(CAPTURE_POINTS):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kurtosis_per_module.png"), dpi=200)
    plt.close(fig)

    return rows


def _iter_articles(rows):
    """Group raw wikitext-2 rows into articles.

    A top-level article heading is ` = Title = `; section headings use two or more
    `=` on each side, so the `= =` test is what separates articles from sections.
    """
    cur = []
    for r in rows:
        s = r.strip()

        # 1. 현재 줄이 새 문서의 최상위 제목인지 확인
        is_article_title = (
            s.startswith("= ") and s.endswith(" =") and not s.startswith("= =")
        )

        # 2. 새 문서 제목이고, 이미 cur에 이전 문서 데이터가 담겨 있다면?
        if is_article_title and cur:
            yield cur # 지금까지 모은 이전 문서(cur)를 밖으로 내보냄
            cur = [] # 버퍼를 비워 새 문서 준비

        cur.append(r) # 현재 줄을 버퍼에 추가
        
    # 3. 파일이 끝나고 남은 마지막 문서 내보내기
    if cur:
        yield cur


def _wikitext_article(index, min_tokens, tokenizer, log):
    """Return one coherent wikitext-2 test article with at least `min_tokens` tokens."""
    import datasets

    rows = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")["test"]["text"]
    long_enough = []
    for article in _iter_articles(rows):
        text = "".join(article)  # rows already carry their trailing newline
        # Cheap prefilter before paying for a real encode: ~4 chars/token is a safe
        # under-estimate for English, so nothing long enough gets discarded here.
        if len(text) < min_tokens * 4:
            continue
        if len(tokenizer(text).input_ids) >= min_tokens:
            long_enough.append(text)
        if len(long_enough) > index:
            break

    if not long_enough or index >= len(long_enough):
        raise ValueError(
            f"--sn_article {index} is out of range: only {len(long_enough)} wikitext-2 "
            f"test articles reach {min_tokens} tokens."
        )
    text = long_enough[index]
    log.info("Analysis text: wikitext-2 test article #{} ({!r}...)".format(
        index, text.strip()[:60]))
    return text


def build_analysis_input(tokenizer, args, log):
    """Tokenize a coherent passage so that position 0 is a genuine [BOS].

    The paper's Figure 2 is explicit that position 0 holds "the massive [BOS] outlier",
    so the input has to actually begin a document. Slicing a window out of the whole
    concatenated corpus (what data_utils.get_wikitext2(eval_mode=True) yields) starts
    mid-sentence and has no BOS at all -- position 0 would still behave as an attention
    sink, but it would be an arbitrary word token, not the thing the paper names.
    """
    #1. 먼저 text 파일로 부터 일관된 내용의 paragraph를 가지고 온다
    if args.sn_text_file:
        with open(args.sn_text_file) as f:
            text = f.read()
        log.info(f"Analysis text: {args.sn_text_file}")
    else:
        text = _wikitext_article(args.sn_article, args.sn_seqlen, tokenizer, log)

    # 2. tokenizer를 처리한 input token 값과 bos의 값을 첨부한다 
    ids = tokenizer(text, return_tensors="pt").input_ids
    bos = tokenizer.bos_token_id
    # Llama-3's fast tokenizer prepends BOS via a post-processor baked into
    # tokenizer.json, which ignores the add_bos_token=False this repo passes -- so it is
    # usually already there. Other tokenizers honour the flag, hence the explicit check.
    if bos is not None and ids[0, 0].item() != bos:
        ids = torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)
    ids = ids[:, : args.sn_seqlen]

    has_bos = bos is not None and ids[0, 0].item() == bos
    if not has_bos:
        log.warning(
            "Tokenizer exposes no bos_token_id; position 0 is an ordinary token. It will "
            "still act as an attention sink, but it is not the paper's [BOS] outlier."
        )
    return ids, has_bos


def build_tokenizer(model_args, training_args):
    common = dict(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    if "Llama-3" in model_args.input_model:
        return PreTrainedTokenizerFast.from_pretrained(**common)
    if "Qwen" in model_args.input_model:
        return AutoTokenizer.from_pretrained(**common)
    return LlamaTokenizerFast.from_pretrained(**common)


def assert_norms_fused(model):
    """Fail loudly if fuse_layer_norms did not run.

    Sn folds its residual renormalization into the existing input_layernorm /
    post_attention_layernorm calls, which is only correct once those norms are
    weightless. ptq_model fuses them under
    `--rotate and (not --deactivate_r1 or --dynamic_residual_scaling)` -- and this script
    enables Sn by toggling model.config directly rather than passing the flag, so
    `--rotate --deactivate_r1` slips through a `--rotate`-only check and then silently
    injects each norm's real per-channel weight into the residual stream. That does not
    crash; it just produces meaningless numbers (Llama-3.1-8B layer 1 went from ~570 to
    ~2e5 excess kurtosis). Check the weights themselves instead of re-deriving the flag
    logic.
    """
    for i, layer in enumerate(model.model.layers):
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            w = getattr(layer, norm_name).weight
            if not torch.allclose(w, torch.ones_like(w), atol=1e-5):
                raise AssertionError(
                    f"layers.{i}.{norm_name} still carries a non-unit weight, so "
                    "fuse_layer_norms did not run and Sn would not be function-"
                    "preserving. Drop --deactivate_r1, or add --dynamic_residual_scaling "
                    "so ptq_model fuses the norms."
                )


def main() -> None:
    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("sn_analysis", ptq_args.eval_out_path)

    # assert ptq_args.rotate, (
    #     "--rotate is required: Sn reuses input_layernorm/post_attention_layernorm as a "
    #     "weightless norm, which only holds after fuse_layer_norms (called under --rotate)."
    # )

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = (
        torch.bfloat16
        if training_args.bf16 or config.torch_dtype == torch.bfloat16
        else torch.float16
    )
    device_map = "auto" if ptq_args.distribute else None

    model_args.net = model_args.input_model.split("/")[-1]
    cls = Qwen2ForCausalLM if "Qwen" in model_args.net else LlamaForCausalLM
    model = cls.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
        device_map=device_map,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    if not ptq_args.distribute:
        model.cuda()

    tokenizer = build_tokenizer(model_args, training_args)
    model = ptq_model(ptq_args, model, log, tokenizer, model_args)
    model.seqlen = training_args.model_max_length
    model.eval()
    model.config.use_cache = False
    assert_norms_fused(model)

    # One fixed sequence, reused verbatim by both arms -- the comparison is only
    # meaningful if the tokens are identical.
    ids, has_bos = build_analysis_input(tokenizer, ptq_args, log)
    inp = ids.to(model.device)
    log.info(
        "Analysis input: {} tokens, position 0 is {}, capture={}".format(
            inp.shape[1], "[BOS]" if has_bos else "an ordinary token", ptq_args.sn_capture
        )
    )

    sub_dir = "keep_first" if ptq_args.sn_keep_first_token else "no_keep_first"
    base_dir = ptq_args.distribution_dir or os.path.join("figures", model_args.net, "Sn")

    out_dir = os.path.join(base_dir, sub_dir)
    os.makedirs(out_dir, exist_ok=True)

    n_layers = len(model.model.layers)
    detail_layers = [i for i in ptq_args.sn_layers if 0 <= i < n_layers]
    if len(detail_layers) != len(ptq_args.sn_layers):
        log.warning(
            "Ignoring out-of-range --sn_layers (model has {} layers)".format(n_layers)
        )

    keep_first = ptq_args.sn_keep_first_token
    kurt_base, detail_base = run_pass(
        model, inp, False, detail_layers, ptq_args.sn_capture, keep_first, log,
        verify_shared=ptq_args.sn_verify_shared,
    )
    kurt_sn, detail_sn = run_pass(
        model, inp, True, detail_layers, ptq_args.sn_capture, keep_first, log,
        verify_shared=ptq_args.sn_verify_shared,
    )

    # (2) kurtosis along the gptq sequential order -- every layer.
    plot_kurtosis_sequential(kurt_base, kurt_sn, n_layers, out_dir)
    log.info("Wrote kurtosis_sequential.png / .csv / kurtosis_per_module.png")

    # (1) 3D surfaces and (3) per-token distributions -- only the requested layers.
    for i in detail_layers:
        layer_dir = os.path.join(out_dir, f"layer{i}")
        os.makedirs(layer_dir, exist_ok=True)
        for name in CAPTURE_POINTS:
            key = (i, name)
            if key not in detail_base:
                continue
            plot_3d(
                detail_base[key], detail_sn[key],
                os.path.join(layer_dir, f"{SHORT[name]}_3d.png"), i, name,
            )
        down = (i, "mlp.down_proj")
        if down in detail_base:
            plot_token_distribution(
                detail_base[down], detail_sn[down], kurt_base[down], kurt_sn[down],
                os.path.join(layer_dir, "down_proj_token_dist.png"),
                i, "mlp.down_proj",
            )
        log.info(f"Wrote figures for layer {i} -> {layer_dir}")

    # Headline number: the paper's claim is specifically about the down_proj input.
    for i in range(n_layers):
        key = (i, "mlp.down_proj")
        if key in kurt_base:
            b, s = kurt_base[key], kurt_sn[key]
            log.info(
                "layer {:>2} down_proj input kurtosis: {:10.1f} -> {:10.1f} ({:+.1f}%)".format(
                    i, b, s, 100.0 * (s - b) / abs(b) if b else float("nan")
                )
            )


if __name__ == "__main__":
    main()
