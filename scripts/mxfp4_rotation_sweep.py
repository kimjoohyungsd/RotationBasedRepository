#!/usr/bin/env python
"""MXFP4 x rotation-group-size sweep (reproduces the paper's Figure 8).

For every (model, rotation group size) it runs the PTQ evaluation with
  * MXFP4 (E2M1 + per-block-32 E8M0) for weights AND activations (W4A4, RTN),
  * a block-diagonal random-Hadamard rotation of the given group size
    (--diagonal --diagonal_size g; R1 fused into weights, R2/R4 applied online as
    Kronecker-structured Walsh-Hadamard transforms of the same block size),
and records WikiText-2 perplexity.

Numbers are written to   logs/MXFP4_RotationGroupSweep/results.csv   (resumable:
already-done rows are skipped) and the figure to
figures/MXFP4_RotationGroupSweep/rotation_group_ppl.png .

Usage:
  python scripts/mxfp4_rotation_sweep.py                 # full sweep + plot
  python scripts/mxfp4_rotation_sweep.py --plot-only      # just (re)draw from CSV
  python scripts/mxfp4_rotation_sweep.py --dry-run        # print commands only
  python scripts/mxfp4_rotation_sweep.py --models Llama-3.1-8B --sizes 32 16
"""
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO, "logs", "MXFP4_RotationGroupSweep")
FIG_DIR = os.path.join(REPO, "figures", "MXFP4_RotationGroupSweep")
CSV_PATH = os.path.join(LOG_DIR, "results.csv")

# net name -> HuggingFace id (must be present in ~/.cache/huggingface)
MODELS = {
    "Llama-2-7b-hf": "meta-llama/Llama-2-7b-hf",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B",
}
SIZES = [1024, 512, 256, 128, 64, 32, 16, 8]  # rotation group sizes (Fig. 8 x-axis)

PPL_RE = re.compile(r"wiki2 ppl is:\s*([0-9.]+)")


def build_cmd(hf_id, g, run_log):
    return [
        sys.executable, os.path.join(REPO, "ptq.py"),
        "--input_model", hf_id,
        "--do_train", "False", "--do_eval", "True",
        "--per_device_eval_batch_size", "4",
        "--model_max_length", "2048",
        "--bf16", "True", "--fp16", "False",
        "--save_safetensors", "False",
        # W4A4 MXFP4 (RTN weights), KV left at 16-bit (W+A scope)
        "--w_bits", "4", "--a_bits", "4", "--k_bits", "16", "--v_bits", "16",
        "--w_rtn", "--mxfp4", "--mx_block", "32",
        "--a_asym",
        # block-diagonal random-Hadamard rotation of size g
        "--rotate", "--diagonal", "--diagonal_size", str(g),
        "--wikitext2",
        "--eval_out_path", run_log,
    ]


def parse_ppl(run_log, stdout):
    for text in (stdout or "", _read(run_log)):
        hits = PPL_RE.findall(text)
        if hits:
            return float(hits[-1])
    return None


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def load_done():
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                if row.get("ppl") not in (None, "", "None"):
                    done[(row["model"], int(row["group_size"]))] = float(row["ppl"])
    return done


def append_row(model, g, ppl):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "group_size", "w_bits", "a_bits", "mxfp4",
                        "mx_block", "ppl", "timestamp"])
        w.writerow([model, g, 4, 4, True, 32,
                    "" if ppl is None else ppl,
                    datetime.datetime.now().isoformat(timespec="seconds")])


def run_sweep(models, sizes, dry_run=False):
    os.makedirs(LOG_DIR, exist_ok=True)
    done = load_done()
    for model in models:
        hf_id = MODELS[model]
        for g in sizes:
            if (model, g) in done:
                print(f"[skip] {model} g={g} (ppl={done[(model, g)]})", flush=True)
                continue
            run_log = os.path.join(LOG_DIR, f"{model}_g{g}.log")
            cmd = build_cmd(hf_id, g, run_log)
            print(f"[run ] {model} g={g}\n       {' '.join(cmd)}", flush=True)
            if dry_run:
                continue
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            ppl = parse_ppl(run_log, proc.stdout + "\n" + proc.stderr)
            if ppl is None:
                print(f"[FAIL] {model} g={g}: no ppl parsed. tail:\n"
                      f"{(proc.stdout + proc.stderr)[-1500:]}", flush=True)
            else:
                print(f"[ ok ] {model} g={g}  ppl={ppl}", flush=True)
            append_row(model, g, ppl)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(CSV_PATH):
        print("No results.csv to plot.")
        return
    data = {}
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("ppl") in (None, "", "None"):
                continue
            data.setdefault(row["model"], {})[int(row["group_size"])] = float(row["ppl"])
    if not data:
        print("results.csv has no finished rows yet.")
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    order = SIZES  # 1024 -> 8, left to right (as in Fig. 8)
    xpos = list(range(len(order)))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors = ["tab:blue", "tab:purple", "tab:orange", "tab:green", "tab:red", "tab:brown"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, (model, d) in enumerate(sorted(data.items())):
        ys = [d.get(g, float("nan")) for g in order]
        ax.plot(xpos, ys, marker=markers[k % len(markers)],
                color=colors[k % len(colors)], label=model)
        # mark this model's optimal (lowest ppl)
        valid = [(i, y) for i, y in enumerate(ys) if y == y]
        if valid:
            bi, by = min(valid, key=lambda t: t[1])
            ax.scatter([bi], [by], color="red", marker="*", s=140, zorder=5)
            ax.annotate(f"{by:.2f}", (bi, by), textcoords="offset points",
                        xytext=(6, -12), color="red", fontsize=8)

    # shade the group size that is optimal on average
    avg = {g: sum(d.get(g, 0) for d in data.values()) /
              max(1, sum(g in d for d in data.values())) for g in order}
    best_g = min((g for g in order if any(g in d for d in data.values())),
                 key=lambda g: avg[g])
    bx = order.index(best_g)
    ax.axvspan(bx - 0.4, bx + 0.4, color="red", alpha=0.10)
    ax.axvline(bx, color="red", ls="--", lw=1)
    ax.annotate(f"Optimal Point\n(Group Size = {best_g})", (bx, ax.get_ylim()[1]),
                color="red", fontsize=9, ha="center", va="top")

    ax.set_xticks(xpos)
    ax.set_xticklabels([str(g) for g in order])
    ax.set_xlabel("Rotation Group Size")
    ax.set_ylabel("Perplexity")
    ax.set_title("Effect of rotation matrix dimension on MXFP4 (W4A4) accuracy")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "rotation_group_ppl.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Figure saved to", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    if not a.plot_only:
        run_sweep(a.models, a.sizes, dry_run=a.dry_run)
    if not a.dry_run:
        plot()


if __name__ == "__main__":
    main()
