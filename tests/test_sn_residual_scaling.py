# coding=utf-8
"""FPTQuant Sn (--dynamic_residual_scaling) must be function-preserving.

    python tests/test_sn_residual_scaling.py

Sn normalizes the residual stream and multiplies each block's output back by the
same per-token scale before the residual add (paper Fig. 2). Without quantization
that is exactly the identity, so a model run with Sn on must produce the same
logits as the same model with Sn off.

The scale is applied at o_proj / down_proj's *input* when those have no bias
(better for quantization: that input is the worst-outlier tensor in the block) and
at the block output otherwise. Both placements are checked -- with a bias, input
placement would silently leave the bias unscaled and break the identity.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402
from eval_utils.modeling_llama import LlamaForCausalLM  # noqa: E402
from utils import fuse_norm_utils  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def build(bias, attn_impl, device):
    cfg = transformers.LlamaConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, vocab_size=64,
        max_position_embeddings=64, attention_bias=bias, mlp_bias=bias,
    )
    cfg._attn_implementation = attn_impl
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(device=device, dtype=torch.float32).eval()
    if bias:  # make the biases non-trivial so an unscaled bias would show up
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.o_proj.bias.normal_(0, 0.5)
                layer.mlp.down_proj.bias.normal_(0, 0.5)
    # Sn reuses the layernorms as weightless norms, which is only valid after fusion
    fuse_norm_utils.fuse_layer_norms(model)
    return model


@torch.no_grad()
def run(model, ids, sn):
    model.config.dynamic_residual_scaling = sn
    return model(input_ids=ids).logits.float()


def test(bias, attn_impl, device):
    model = build(bias, attn_impl, device)
    ids = torch.randint(0, 64, (2, 16), device=device)
    off = run(model, ids, False)
    on = run(model, ids, True)
    err = (on - off).abs().max().item()
    scale = off.abs().max().item()
    tag = f"bias={bias}, attn={attn_impl}"
    check(f"Sn is function-preserving ({tag})", err < 1e-3 * max(scale, 1.0),
          f"max abs logit diff = {err:.2e} (logit scale {scale:.2e})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"FPTQuant Sn function preservation ({device})\n")
    for attn_impl in ("eager", "sdpa"):
        for bias in (False, True):
            test(bias, attn_impl, device)
    print("\nRESULT:", "PASS" if not FAILED else f"FAIL ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
