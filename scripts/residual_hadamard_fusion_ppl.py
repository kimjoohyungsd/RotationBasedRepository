#!/usr/bin/env python
"""W4A4 MXFP4 WikiText-2 PPL with a block-diagonal random-Hadamard rotation of
the MXFP4 group size (32) applied to the residual path, comparing
fuse_layer_norms applied vs. not -- i.e. scripts/no_rotation_fusion_ppl.py's
LayerNorm-fusion confound isolation, but now *with* the rotation on instead of
--rotate OFF the whole time.

Same recipe as scripts/mxfp4_rotation_sweep.py at rotation group size g=32
(MXFP4 (E2M1 + per-block-32 E8M0) weights AND activations, W4A4 RTN, block-
diagonal random Hadamard: R1 fused into weights, R2/R4 online as Kronecker-
structured Walsh-Hadamard transforms of the same block size), the ONLY new
knob being whether utils.fuse_norm_utils.fuse_layer_norms runs first:

  * fused=True  -- the normal pipeline. eval_utils.main.ptq_model always calls
                   fuse_layer_norms under --rotate (R1 needs the RMSNorm gain
                   folded into the downstream linears to stay function-
                   preserving).
  * fused=False -- fuse_layer_norms monkey-patched to a no-op, so R1 rotates
                   the embedding / q,k,v / o / gate,up / down / lm_head weights
                   while every RMSNorm gamma is left in the ORIGINAL basis.
                   NOT function-preserving -- that residual-path basis mismatch
                   is exactly the quantity this script measures the PPL cost of.

PPL itself comes from simple_wikitext2_ppl (a direct whole-model forward per
window), identical to no_rotation_fusion_ppl.py, so the rows here line up
one-to-one with logs/NoRotationFusionPPL/results.csv.

    python scripts/residual_hadamard_fusion_ppl.py
    python scripts/residual_hadamard_fusion_ppl.py --models Llama-2-7b-hf
    python scripts/residual_hadamard_fusion_ppl.py --fused 0   # unfused only (parallel launch)

Results appended to logs/ResidualHadamardFusionPPL/results.csv (resumable).
"""
import argparse
import csv
import datetime
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
LOG_DIR = os.path.join(REPO, "logs", "ResidualHadamardFusionPPL")
CSV_PATH = os.path.join(LOG_DIR, "results.csv")

DIAGONAL_SIZE = 32  # block-diagonal random-Hadamard block == MXFP4 group size

MODELS = {
    "Llama-2-7b-hf": ("meta-llama/Llama-2-7b-hf", "llama"),
    "Llama-3.1-8B": ("meta-llama/Llama-3.1-8B", "llama"),
    "Qwen3-8B": ("Qwen/Qwen3-8B", "qwen3"),
}


def simple_wikitext2_ppl(model, testenc, seqlen, dev, batch_size=1):
    """Standard causal-LM WikiText-2 perplexity via a direct whole-model
    forward per window -- byte-for-byte the same function as
    scripts/no_rotation_fusion_ppl.py, so the numbers are directly
    comparable."""
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


# scripts/mxfp4_rotation_sweep.py's g=32 recipe: same W4A4 MXFP4 RTN flags as
# no_rotation_fusion_ppl.py, plus the block-diagonal random-Hadamard rotation.
BASE_FLAGS = [
    "--do_train", "False", "--do_eval", "True",
    "--per_device_eval_batch_size", "4",
    "--model_max_length", "2048",
    "--bf16", "True", "--fp16", "False",
    "--save_safetensors", "False",
    "--w_bits", "4", "--a_bits", "4", "--k_bits", "16", "--v_bits", "16",
    "--w_rtn", "--mxfp4", "--mx_block", "32",
    "--a_asym",
    "--rotate", "--diagonal", "--diagonal_size", str(DIAGONAL_SIZE),
    "--rotate_mode", "hadamard",
]


def run_one(model_name, hf_id, arch, fuse: bool, run_log):
    argv = ["ptq.py", "--input_model", hf_id, "--eval_out_path", run_log] + BASE_FLAGS
    old_argv = sys.argv
    sys.argv = argv
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
        from utils.process_args import process_args_ptq
        from utils import fuse_norm_utils, data_utils, utils as rbr_utils
        from eval_utils.main import ptq_model

        model_args, training_args, ptq_args = process_args_ptq()
        assert ptq_args.rotate, "this script measures the residual-Hadamard case"

        config = transformers.AutoConfig.from_pretrained(model_args.input_model)
        if arch == "qwen3":
            from eval_utils.modeling_qwen3 import Qwen3ForCausalLM as ModelClass
        else:
            from eval_utils.modeling_llama import LlamaForCausalLM as ModelClass
        model = ModelClass.from_pretrained(model_args.input_model, config=config,
                                           torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.config.dynamic_residual_scaling = False
        model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_args.input_model, use_fast=True)

        log = rbr_utils.get_logger("residual_hadamard_fusion_ppl", ptq_args.eval_out_path)

        if not fuse:
            # eval_utils.main calls fuse_norm_utils.fuse_layer_norms(model) by
            # attribute at run time, so replacing the attribute here is enough
            # to make the rotation run with the RMSNorm gains left untouched.
            fuse_norm_utils.fuse_layer_norms = lambda *a, **k: log.info(
                "fuse_layer_norms SKIPPED (rotate=True, RMSNorm gains left in original basis)"
            )
            log.info("residual-path RMSNorm weight fusion: OFF")
        else:
            log.info("residual-path RMSNorm weight fusion: ON (normal --rotate pipeline)")

        model = ptq_model(ptq_args, model, log, tokenizer, model_args)
        model.seqlen = training_args.model_max_length

        testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed, seqlen=model.seqlen, tokenizer=tokenizer, eval_mode=True,
        )
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
            w.writerow(["model", "fused", "rotation", "diagonal_size", "w_bits",
                        "a_bits", "mxfp4", "mx_block", "ppl", "timestamp"])
        w.writerow([model, fused, "hadamard_blockdiag", DIAGONAL_SIZE, 4, 4, True, 32,
                    "" if ppl is None else ppl,
                    datetime.datetime.now().isoformat(timespec="seconds")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--fused", choices=["0", "1", "both"], default="both",
                    help="run just the unfused (0), just the fused (1), or both (default) "
                         "-- lets you launch the two variants as separate parallel "
                         "processes on different GPUs.")
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
                import traceback
                traceback.print_exc()
                print(f"[FAIL] {model_name} fused={fused}: {e}", flush=True)
                ppl = None
            append_row(model_name, fused, ppl)


if __name__ == "__main__":
    main()
