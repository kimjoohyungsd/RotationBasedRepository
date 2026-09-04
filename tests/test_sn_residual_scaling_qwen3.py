# coding=utf-8
"""FPTQuant Sn (--dynamic_residual_scaling) must be function-preserving on Qwen3.

    python tests/test_sn_residual_scaling_qwen3.py

Same check as tests/test_sn_residual_scaling.py, but for eval_utils/modeling_qwen3.py's
ported residual_scale threading. Sn normalizes the residual stream and multiplies each
block's output back by the same per-token scale before the residual add; without
quantization that is exactly the identity, so Sn-on logits must equal Sn-off logits.

Qwen3 has no linear biases (config.attention_bias defaults False), so the scale is
applied at o_proj / down_proj's input in every layer. Qwen3's per-head q_norm/k_norm
are off the residual path and must be left untouched -- covered implicitly here: a
q_norm/k_norm that consumed or emitted the running scale would break the identity.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402
from eval_utils.modeling_qwen3 import Qwen3ForCausalLM  # noqa: E402
from utils import fuse_norm_utils  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def build(attn_impl, device, dtype):
    cfg = transformers.models.qwen3.configuration_qwen3.Qwen3Config(
        hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=64,
        max_position_embeddings=64, sliding_window=None, tie_word_embeddings=False,
    )
    cfg._attn_implementation = attn_impl
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    # make the RMSNorm gains non-trivial so fusion actually moves something
    with torch.no_grad():
        for m in model.modules():
            if m.__class__.__name__ == "Qwen3RMSNorm":
                m.weight.normal_(1.0, 0.1)
    # Sn reuses the residual-path layernorms as weightless norms -> valid only after fusion
    fuse_norm_utils.fuse_layer_norms(model)
    return model


@torch.no_grad()
def run(model, ids, sn):
    model.config.dynamic_residual_scaling = sn
    return model(input_ids=ids).logits.float()


def test(attn_impl, device, dtype):
    model = build(attn_impl, device, dtype)
    ids = torch.randint(0, 64, (2, 16), device=device)
    off = run(model, ids, False)
    on = run(model, ids, True)
    err = (on - off).abs().max().item()
    scale = off.abs().max().item()
    tol = (2e-3 if dtype == torch.float16 else 1e-3) * max(scale, 1.0)
    tag = f"attn={attn_impl}, dtype={str(dtype).split('.')[-1]}"
    check(f"Sn is function-preserving ({tag})", err < tol,
          f"max abs logit diff = {err:.2e} (logit scale {scale:.2e}, tol {tol:.2e})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FPTQuant Sn function preservation on Qwen3 ({device})\n")
    for attn_impl in ("eager", "sdpa"):
        for dtype in (torch.float32, torch.float16):
            if dtype == torch.float16 and device == "cpu":
                continue
            test(attn_impl, device, dtype)
    print("\nRESULT:", "PASS" if not FAILED else f"FAIL ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
