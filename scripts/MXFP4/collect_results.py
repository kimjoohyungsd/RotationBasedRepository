#!/usr/bin/env python
"""Print a WikiText-2 PPL table for the MXFP4 GPTQ experiments (parsed from logs/MXFP4/*).

    python scripts/MXFP4/collect_results.py [--methods GPTQ BRQ-GPTQ SmoothQuant_GPTQ]

SmoothQuant_GPTQ is reported at its best alpha (alpha in parentheses); every alpha is listed below the table.
"""
import argparse
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(REPO, "logs", "MXFP4")
MODELS = [
    "Llama-2-7b-hf", "Llama-2-13b-hf", "Meta-Llama-3-8B", "Qwen3-8B", "Qwen3-14B",
]
PPL_RE = re.compile(r"wiki2 ppl is: ([0-9.eE+-]+)")


def read_ppl(path):
    try:
        m = PPL_RE.findall(open(path).read())
    except OSError:
        return None
    return float(m[-1]) if m else None


def method_results(method, model):
    """-> (best_ppl, tag, {alpha: ppl})"""
    if method == "SmoothQuant_GPTQ":
        per_alpha = {}
        for d in glob.glob(os.path.join(LOGS, method, model, "alpha_*")):
            for f in glob.glob(os.path.join(d, "log_*.txt")):
                p = read_ppl(f)
                if p is not None:
                    per_alpha[d.rsplit("alpha_", 1)[1]] = p
        if not per_alpha:
            return None, "", {}
        a = min(per_alpha, key=per_alpha.get)
        return per_alpha[a], f" (a={a})", per_alpha
    for f in glob.glob(os.path.join(LOGS, method, model, "log_*.txt")):
        p = read_ppl(f)
        if p is not None:
            return p, "", {}
    return None, "", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["GPTQ", "BRQ-GPTQ", "SmoothQuant_GPTQ"])
    args = ap.parse_args()

    rows, alphas = [], []
    for model in MODELS:
        row = [model]
        for m in args.methods:
            p, tag, per_alpha = method_results(m, model)
            row.append("-" if p is None else f"{p:.3f}{tag}")
            if per_alpha:
                alphas.append((model, per_alpha))
        rows.append(row)

    header = ["Model (W4A4 MXFP4, WikiText-2 PPL)"] + args.methods
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    print(line(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(line(r))
    for model, per_alpha in alphas:
        print(f"\n{model} SmoothQuant_GPTQ alpha sweep: " +
              ", ".join(f"{a}: {p:.3f}" for a, p in sorted(per_alpha.items())))


if __name__ == "__main__":
    main()
