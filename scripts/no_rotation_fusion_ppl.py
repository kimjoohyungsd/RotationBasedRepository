#!/usr/bin/env python
"""W4A4 MXFP4 WikiText-2 PPL, with --rotate OFF the whole time, comparing
fuse_layer_norms applied vs. not -- isolating exactly the confound found in
growth_rate_analysis.py (Llama-2-7b-hf layer 10's post_attention_layernorm
gamma sits at mean 0.237 across every channel) at the level that actually
matters: does it change the final quantized PPL, not just what a hook sees.

Reuses eval_utils.main.ptq_model (weight RTN + MXFP4 activation quant, same
code ptq.py itself calls) verbatim -- the only new behavior is calling
utils.fuse_norm_utils.fuse_layer_norms(model) by hand before ptq_model when
requested, since ptq_model only ever fuses when --rotate (or
--dynamic_residual_scaling, which brings its own extra behavior) is on.
PPL itself comes from simple_wikitext2_ppl below (a direct whole-model
forward per window) rather than utils.eval_utils.evaluator's Catcher/
layer-by-layer loop, which hit two Qwen3 x transformers-4.57 incompatibilities
in a row (one patched in utils/eval_utils.py along the way, see git diff) --
both compute the exact same standard causal-LM PPL over the same windowed
dataset, so Llama numbers from either path are directly comparable.

    python scripts/no_rotation_fusion_ppl.py
    python scripts/no_rotation_fusion_ppl.py --models Llama-2-7b-hf

Results appended to logs/NoRotationFusionPPL/results.csv (resumable).
"""
import argparse
import csv
import datetime
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
LOG_DIR = os.path.join(REPO, "logs", "NoRotationFusionPPL")
CSV_PATH = os.path.join(LOG_DIR, "results.csv")

MODELS = {
    "Llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "llama"),
    "Llama-3.1-8B": ("meta-llama/Llama-3.1-8B", "llama"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "qwen3"),
}


def simple_wikitext2_ppl(model, testenc, seqlen, dev, batch_size=1):
    """Standard causal-LM WikiText-2 perplexity via a direct whole-model
    forward per window -- mathematically the same quantity
    utils.eval_utils.evaluator computes (both are cross-entropy over the same
    windowed dataset; that one just runs one decoder layer across the whole
    batch at a time as a memory optimization), without depending on its
    Catcher/layer-by-layer loop's transformers-version assumptions."""
    import torch
    import torch.nn as nn

    input_ids = testenc.input_ids
    nsamples = input_ids.numel() // seqlen
    input_ids = input_ids[:, :nsamples * seqlen].view(nsamples, seqlen)

    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    with torch.no_grad():
        for i in range(0, nsamples, batch_size):
            batch = input_ids[i:i + batch_size].to(dev)
            logits = model(batch).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                            shift_labels.reshape(-1))
            nlls.append(loss.float() * shift_labels.numel())
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seqlen - 1)))
    return ppl.item()

# Same W4A4 MXFP4 RTN recipe scripts/mxfp4_rotation_sweep.py uses for its
# (already-published) rotated numbers, minus --rotate, so the "fused"/
# "unfused" pair here is directly comparable to that sweep's --rotate rows.
BASE_FLAGS = [
    "--do_train", "False", "--do_eval", "True",
    "--per_device_eval_batch_size", "4",
    "--model_max_length", "2048",
    "--bf16", "True", "--fp16", "False",
    "--save_safetensors", "False",
    "--w_bits", "4", "--a_bits", "4", "--k_bits", "16", "--v_bits", "16",
    "--w_rtn", "--mxfp4", "--mx_block", "32",
    "--a_asym",
]


def run_one(model_name, hf_id, arch, fuse: bool, run_log):
    argv = ["ptq.py", "--input_model", hf_id, "--eval_out_path", run_log] + BASE_FLAGS
    old_argv = sys.argv
    sys.argv = argv
    try:
        # Imported here, after sys.argv is set and with REPO on sys.path --
        # process_args_ptq() reads sys.argv directly (see utils/process_args.py).
        import torch
        import transformers
        from transformers import AutoTokenizer
        from utils.process_args import process_args_ptq
        from utils import fuse_norm_utils, data_utils, utils as rbr_utils
        from eval_utils.main import ptq_model

        model_args, training_args, ptq_args = process_args_ptq()
        assert not ptq_args.rotate, "this script compares the no-rotation case only"

        config = transformers.AutoConfig.from_pretrained(model_args.input_model)
        if arch == "qwen3":
            from eval_utils.modeling_qwen3 import Qwen3ForCausalLM as ModelClass
        else:
            from eval_utils.modeling_llama import LlamaForCausalLM as ModelClass
        model = ModelClass.from_pretrained(model_args.input_model, config=config,
                                           torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.config.dynamic_residual_scaling = False  # see growth_rate_analysis.py's load_model
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_args.input_model, use_fast=True)

        log = rbr_utils.get_logger("no_rotation_fusion_ppl", ptq_args.eval_out_path)

        if fuse:
            fuse_norm_utils.fuse_layer_norms(model)
            log.info("fuse_layer_norms applied by hand (rotate=False)")

        model = ptq_model(ptq_args, model, log, tokenizer, model_args)
        model.seqlen = training_args.model_max_length

        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed, seqlen=model.seqlen, tokenizer=tokenizer, eval_mode=True,
        )
        # utils.eval_utils.evaluator's Catcher/layer-by-layer loop (a memory
        # optimization, not a different formula) hits two Qwen3 x transformers
        # 4.57 incompatibilities in a row (missing .attention_type -- patched
        # above in utils/eval_utils.py -- then a None-unpacking error one step
        # further in); a direct whole-model forward per window computes the
        # exact same standard causal-LM PPL without depending on either.
        ppl = simple_wikitext2_ppl(model, testloader, model.seqlen, rbr_utils.DEV)
        log.info(f"wiki2 ppl is: {ppl}")

        del model
        torch.cuda.empty_cache()
        return float(ppl)
    finally:
        sys.argv = old_argv


def load_done():
    done = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                if row.get("ppl") not in (None, "", "None"):
                    done[(row["model"], row["fused"])] = float(row["ppl"])
    return done


def append_row(model, fused, ppl):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "fused", "w_bits", "a_bits", "mxfp4", "mx_block",
                       "ppl", "timestamp"])
        w.writerow([model, fused, 4, 4, True, 32,
                   "" if ppl is None else ppl,
                   datetime.datetime.now().isoformat(timespec="seconds")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--fused", choices=["0", "1", "both"], default="both",
                    help="run just the unfused (0), just the fused (1), or both variants "
                         "(default) -- lets you launch the two variants of one model as "
                         "separate parallel processes on different GPUs.")
    args = ap.parse_args()
    fused_values = {"0": [False], "1": [True], "both": [False, True]}[args.fused]

    os.makedirs(LOG_DIR, exist_ok=True)
    done = load_done()
    for model_name in args.models:
        hf_id, arch = MODELS[model_name]
        for fused in fused_values:
            key = (model_name, str(fused))
            if args.skip_existing and key in done:
                print(f"[skip] {model_name} fused={fused} (ppl={done[key]})", flush=True)
                continue
            run_log = os.path.join(LOG_DIR, f"{model_name}_fused-{fused}.log")
            print(f"[run ] {model_name} fused={fused}", flush=True)
            try:
                ppl = run_one(model_name, hf_id, arch, fused, run_log)
                print(f"[ ok ] {model_name} fused={fused}  ppl={ppl}", flush=True)
            except Exception as e:
                print(f"[FAIL] {model_name} fused={fused}: {e}", flush=True)
                ppl = None
            append_row(model_name, fused, ppl)


if __name__ == "__main__":
    main()
