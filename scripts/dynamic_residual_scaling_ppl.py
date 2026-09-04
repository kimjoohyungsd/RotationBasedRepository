#!/usr/bin/env python
"""W4A4 MXFP4 WikiText-2 PPL with a block-diagonal random-Hadamard rotation of
the MXFP4 group size (32) applied to R1, R2 and R4, comparing FPTQuant's Sn
transform (--dynamic_residual_scaling) ON vs. OFF -- the ONLY knob varied.

Same recipe as scripts/mxfp4_rotation_sweep.py at rotation group size g=32
(MXFP4 E2M1 + per-block-32 E8M0 for weights AND activations, W4A4 RTN,
--rotate --diagonal --diagonal_size 32 --rotate_mode hadamard: R1 fused into
the weights, R2/R4 online as Kronecker Walsh-Hadamard transforms of the same
block size). Under --rotate with R1 active, eval_utils.main.ptq_model runs
fuse_layer_norms in BOTH variants, so the Sn-on / Sn-off pair differs only by
the per-token residual renormalization FPTQuant Sn threads through the decoder
stack (arXiv:2506.04985 Sec. 3.1.3) -- function-preserving without
quantization, outlier-reducing at o_proj/down_proj inputs with it.

PPL comes from simple_wikitext2_ppl (a direct whole-model forward per window),
identical to scripts/no_rotation_fusion_ppl.py / residual_hadamard_fusion_ppl.py,
so these rows line up one-to-one with logs/NoRotationFusionPPL/ and
logs/ResidualHadamardFusionPPL/.

NOTE: FPTQuant Sn is implemented in eval_utils/modeling_llama.py only. The
Qwen3 decoder (eval_utils/modeling_qwen3.py) has the same residual_scale
threading ported in for this experiment -- see
tests/test_sn_residual_scaling_qwen3.py.

    python scripts/dynamic_residual_scaling_ppl.py
    python scripts/dynamic_residual_scaling_ppl.py --models Llama-2-7b-hf
    python scripts/dynamic_residual_scaling_ppl.py --sn 1        # Sn-on only (parallel launch)

Results appended to logs/DynamicResidualScalingPPL/results.csv (resumable).
"""
import argparse
import csv
import datetime
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
LOG_DIR = os.path.join(REPO, "logs", "DynamicResidualScalingPPL")
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


# scripts/mxfp4_rotation_sweep.py's g=32 recipe (R1 + R2 + R4 all active).
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


def run_one(model_name, hf_id, arch, sn: bool, run_log):
    argv = ["ptq.py", "--input_model", hf_id, "--eval_out_path", run_log] + BASE_FLAGS
    if sn:
        argv.append("--dynamic_residual_scaling")
    old_argv = sys.argv
    sys.argv = argv
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
        from utils.process_args import process_args_ptq
        from utils import data_utils, utils as rbr_utils
        from eval_utils.main import ptq_model

        model_args, training_args, ptq_args = process_args_ptq()
        assert ptq_args.rotate, "this script measures the rotated (R1+R2+R4) case"
        assert ptq_args.dynamic_residual_scaling == sn

        config = transformers.AutoConfig.from_pretrained(model_args.input_model)
        if arch == "qwen3":
            from eval_utils.modeling_qwen3 import Qwen3ForCausalLM as ModelClass
        else:
            from eval_utils.modeling_llama import LlamaForCausalLM as ModelClass
        # bf16 (not fp16): the Sn per-token residual scale threaded across R1/R2/R4
        # Hadamard rotation + MXFP4 on a deep (36-layer) model can push a scaled
        # o_proj/down_proj input past fp16's ~6.5e4 range -> Inf -> NaN -> a
        # device-side assert in the MXFP4 exponent lookup. bf16 has fp32's 8-bit
        # exponent, so it holds; it also matches --bf16 True in BASE_FLAGS.
        model = ModelClass.from_pretrained(model_args.input_model, config=config,
                                           torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        # ptq_model sets model.config.dynamic_residual_scaling from ptq_args; set a
        # default now so the modeling code's forward never dereferences a missing attr.
        model.config.dynamic_residual_scaling = False

        # Keep the model CPU-resident through ptq_model. utils.hadamard_utils.
        # apply_exact_had_to_linear (the --diagonal R1/R2/R4 path) streams each
        # weight to the GPU one tensor at a time for the float64 Hadamard matmul
        # and writes it back to its original (CPU) device, so rotation still runs
        # on the GPU while never holding more than one big tensor there. For an
        # 8B model with a 128k-row embedding / lm_head that float64 temporary is
        # ~4 GiB and OOMs a 24 GiB card when the whole model is also resident.
        # RTN weight quant then runs per-layer on CPU (gptq_utils.rtn_fwrd honours
        # each layer's device); the quantized fp16 model is moved to the GPU once,
        # below, for the PPL forward (~16 GiB, fits).

        tokenizer = AutoTokenizer.from_pretrained(model_args.input_model, use_fast=True)

        log = rbr_utils.get_logger("dynamic_residual_scaling_ppl", ptq_args.eval_out_path)
        log.info(f"dynamic_residual_scaling (FPTQuant Sn): {'ON' if sn else 'OFF'}")

        model = ptq_model(ptq_args, model, log, tokenizer, model_args)
        model.cuda()
        assert bool(model.config.dynamic_residual_scaling) == sn, \
            "ptq_model did not propagate the Sn flag as expected"
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
                    done[(row["model"], row["sn"])] = float(row["ppl"])
    return done


def append_row(model, sn, ppl):
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "sn", "rotation", "diagonal_size", "w_bits",
                        "a_bits", "mxfp4", "mx_block", "ppl", "timestamp"])
        w.writerow([model, sn, "hadamard_blockdiag_R1R2R4", DIAGONAL_SIZE, 4, 4, True, 32,
                    "" if ppl is None else ppl,
                    datetime.datetime.now().isoformat(timespec="seconds")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--sn", choices=["0", "1", "both"], default="both",
                    help="run just Sn-off (0), just Sn-on (1), or both (default) -- lets "
                         "you launch the two variants as separate parallel processes on "
                         "different GPUs.")
    args = ap.parse_args()
    sn_values = {"0": [False], "1": [True], "both": [False, True]}[args.sn]

    os.makedirs(LOG_DIR, exist_ok=True)
    done = load_done()
    for model_name in args.models:
        hf_id, arch = MODELS[model_name]
        for sn in sn_values:
            key = (model_name, str(sn))
            if args.skip_existing and key in done:
                print(f"[skip] {model_name} sn={sn} (ppl={done[key]})", flush=True)
                continue
            run_log = os.path.join(LOG_DIR, f"{model_name}_sn-{sn}.log")
            print(f"[run ] {model_name} sn={sn}", flush=True)
            try:
                ppl = run_one(model_name, hf_id, arch, sn, run_log)
                print(f"[ ok ] {model_name} sn={sn}  ppl={ppl}", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[FAIL] {model_name} sn={sn}: {e}", flush=True)
                ppl = None
            append_row(model_name, sn, ppl)


if __name__ == "__main__":
    main()
