# coding=utf-8
"""Integration check: the LieReSpinQuant forward path agrees with ReSpinQuant.

    python tests/test_lie_integration.py

The Lie path is only a *reparameterization* of the residual bases -- given the
same bases it must compute the same thing as the existing (already validated)
ReSpinQuant path.  This builds one tiny model, runs it with a Lie chain, then
materialises that chain into dense R1/R2/R1_final RotateModules and runs the
ReSpinQuant path over the identical weights.  The logits must match.

It also checks that a chain with all gates zeroed reduces exactly to a single
global rotation (plain SpinQuant), which is the intended initialisation.
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402

from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant  # noqa: E402
from utils.hadamard_utils import hadamard_matrix, random_hadamard_matrix  # noqa: E402
from utils.lie_rotation import LieBasisChain  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)
    return ok


class RotateModule(nn.Module):
    """Same holder optimize_rotation.py uses."""

    def __init__(self, R_init):
        super().__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32))


def tiny_model(device, dtype=torch.float32, hidden=128, layers=2, heads=4):
    cfg = transformers.LlamaConfig(
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        vocab_size=64,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = LlamaForCausalLMQuant(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model


def install_lie(model, rank, gate_init, device):
    cfg = model.config
    hidden = cfg.hidden_size
    head_dim = hidden // cfg.num_attention_heads
    cfg.respinquant = True
    cfg.lierespinquant = True
    torch.manual_seed(7)
    chain = LieBasisChain(
        dim=hidden,
        num_layers=cfg.num_hidden_layers,
        rank=rank,
        base=hadamard_matrix(hidden, device),
        gate_init=gate_init,
        device=device,
    ).to(device)
    model.model.lie_chain = chain
    for i in range(cfg.num_hidden_layers):
        model.model.layers[i].self_attn.R2 = RotateModule(
            random_hadamard_matrix(head_dim, device)).to(device)
    return chain


def freeze_chain_into_dense(model, chain):
    """Replace the Lie chain by the dense bases it currently produces."""
    with torch.no_grad():
        bases = chain.bases()
    cfg = model.config
    for i in range(cfg.num_hidden_layers):
        model.model.layers[i].R1 = RotateModule(bases[2 * i].clone())
        model.model.layers[i].R2 = RotateModule(bases[2 * i + 1].clone())
    model.model.R1_final = RotateModule(bases[-1].clone())
    model.model.lie_chain = None
    cfg.lierespinquant = False


@torch.no_grad()
def logits_of(model, ids):
    return model(input_ids=ids).logits.float()


def test_lie_matches_respinquant(device):
    model = tiny_model(device)
    chain = install_lie(model, rank=8, gate_init=0.3, device=device)
    ids = torch.randint(0, 64, (2, 16), device=device)

    lie_logits = logits_of(model, ids)
    ranks = chain.effective_ranks()

    freeze_chain_into_dense(model, chain)
    respin_logits = logits_of(model, ids)

    err = (lie_logits - respin_logits).abs().max().item()
    scale = respin_logits.abs().max().item()
    check("Lie forward == ReSpinQuant forward on identical bases",
          err < 1e-3 * max(scale, 1.0),
          f"max abs diff = {err:.2e} (logit scale {scale:.2e})")
    check("gates are all open at gate_init=0.3", all(r == 8 for r in ranks),
          f"effective ranks = {ranks}")


def test_zero_gates_is_single_rotation(device):
    """gamma = 0 -> every dR = I -> all bases equal, i.e. plain SpinQuant."""
    model = tiny_model(device)
    chain = install_lie(model, rank=8, gate_init=0.0, device=device)
    with torch.no_grad():
        for d in chain.deltas:
            d.gamma.zero_()
        bases = chain.bases()
    spread = max((B - bases[0]).abs().max().item() for B in bases)
    check("zero gates collapse the chain to one global rotation", spread == 0.0,
          f"max deviation across bases = {spread:.2e}")

    ids = torch.randint(0, 64, (2, 16), device=device)
    lie_logits = logits_of(model, ids)
    check("forward is finite with identity transitions",
          torch.isfinite(lie_logits).all().item())


def test_parameter_budget(device):
    model = tiny_model(device, hidden=128, layers=2)
    chain = install_lie(model, rank=8, gate_init=0.1, device=device)
    n_lie = sum(p.numel() for p in chain.trainable_parameters())
    cfg = model.config
    n_dense = (2 * cfg.num_hidden_layers + 1) * cfg.hidden_size ** 2
    check("Lie chain is far smaller than dense ReSpinQuant bases", n_lie < n_dense,
          f"{n_lie} vs {n_dense} ({n_dense / n_lie:.1f}x fewer)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"LieReSpinQuant integration ({device})\n")
    print("[1] equivalence with the ReSpinQuant path")
    test_lie_matches_respinquant(device)
    print("[2] identity initialisation")
    test_zero_gates_is_single_rotation(device)
    print("[3] parameter budget")
    test_parameter_budget(device)
    print("\nRESULT:", "PASS" if not FAILED else f"FAIL ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
