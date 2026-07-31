# coding=utf-8
"""Correctness checks for the low-rank Cayley residual transitions.

    python tests/test_lie_rotation.py

Verifies the four properties the method rests on:
  1. the Woodbury low-rank form equals the dense Cayley transform,
  2. every transition is exactly orthogonal *and* in SO(D) (det = +1) with no
     determinant correction -- the gap ReSpinQuant's `U_sub @ Vhs` leaves open,
  3. applying the transition in O(D r) matches multiplying by the dense matrix,
  4. the recursively built bases are orthogonal and satisfy
     `bases[k]^T bases[k+1] == deltas[k]` exactly, so the residual transition
     the model pays for at inference *is* the learned low-rank rotation.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.hadamard_utils import hadamard_matrix  # noqa: E402
from utils.lie_rotation import (  # noqa: E402
    LieBasisChain,
    LowRankCayley,
    cayley_dense,
    cayley_lowrank_factors,
)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)
    return ok


def max_err(a, b):
    return (a - b).abs().max().item()


def test_woodbury_matches_dense(D=256, r=16, device="cpu"):
    torch.manual_seed(0)
    m = LowRankCayley(D, r, gate_init=0.5, dtype=torch.float64, device=device)
    A = m.generator()
    check("generator is skew-symmetric", max_err(A, -A.t()) < 1e-12,
          f"max|A + A^T| = {max_err(A, -A.t()):.2e}")

    dense = cayley_dense(A)
    fast = m.matrix()
    e = max_err(dense, fast)
    check("low-rank Cayley == dense Cayley", e < 1e-9, f"max abs diff = {e:.2e}")


def test_orthogonal_and_special(D=256, r=16, device="cpu"):
    torch.manual_seed(1)
    for gate in (1e-3, 0.5, 5.0):
        m = LowRankCayley(D, r, gate_init=gate, dtype=torch.float64, device=device)
        dR = m.matrix()
        I = torch.eye(D, dtype=dR.dtype)
        e = max_err(dR @ dR.t(), I)
        sign, logdet = torch.linalg.slogdet(dR)
        check(f"orthogonal (gate_init={gate})", e < 1e-9, f"max|dR dR^T - I| = {e:.2e}")
        check(f"det = +1, i.e. in SO(D) (gate_init={gate})",
              sign.item() > 0 and abs(logdet.item()) < 1e-8,
              f"sign={sign.item():.0f} logdet={logdet.item():.2e}")


def test_apply_right(D=512, r=32, device="cpu"):
    torch.manual_seed(2)
    m = LowRankCayley(D, r, gate_init=0.3, dtype=torch.float64, device=device)
    x = torch.randn(7, 13, D, dtype=torch.float64)
    ref = x @ m.matrix()
    got = m.apply_right(x)
    e = max_err(ref, got)
    check("O(Dr) apply_right == x @ dR", e < 1e-10, f"max abs diff = {e:.2e}")


def test_identity_at_zero_gate(D=128, r=8):
    m = LowRankCayley(D, r, gate_init=0.0, dtype=torch.float64)
    with torch.no_grad():
        m.gamma.zero_()
    e = max_err(m.matrix(), torch.eye(D, dtype=torch.float64))
    check("gamma = 0 gives dR = I exactly (degenerates to SpinQuant)", e == 0.0,
          f"max abs diff = {e:.2e}")


def test_basis_chain(D=256, L=4, r=16):
    torch.manual_seed(3)
    base = hadamard_matrix(D, "cpu").to(torch.float64)
    chain = LieBasisChain(D, L, r, base, gate_init=0.4)
    for d in chain.deltas:
        d.to(torch.float64)
    chain.base = chain.base.to(torch.float64)

    factors = chain.transition_factors()
    bases = chain.bases(factors)
    check("chain produces 2L+1 bases", len(bases) == 2 * L + 1, f"got {len(bases)}")

    I = torch.eye(D, dtype=torch.float64)
    worst_orth = max(max_err(B @ B.t(), I) for B in bases)
    check("every basis is orthogonal", worst_orth < 1e-8,
          f"max|B B^T - I| = {worst_orth:.2e}")

    worst_t = 0.0
    for k, d in enumerate(chain.deltas):
        T = bases[k].t() @ bases[k + 1]     # what the residual stream must apply
        worst_t = max(worst_t, max_err(T, d.matrix()))
    check("bases[k]^T bases[k+1] == deltas[k] (no post-hoc approximation)",
          worst_t < 1e-8, f"max abs diff = {worst_t:.2e}")

    # the residual correction installed at inference: residual + (residual P) Z P^T
    x = torch.randn(3, 11, D, dtype=torch.float64)
    P, Z = factors[0]
    got = x + ((x @ P) @ Z) @ P.t()
    ref = x @ (bases[0].t() @ bases[1])
    e = max_err(got, ref)
    check("inference-time low-rank residual == exact basis transition", e < 1e-9,
          f"max abs diff = {e:.2e}")


def test_gradients(D=128, r=8):
    torch.manual_seed(4)
    m = LowRankCayley(D, r, gate_init=1e-2)
    x = torch.randn(4, D)
    loss = m.apply_right(x).pow(2).sum()
    loss.backward()
    grads = {n: p.grad for n, p in m.named_parameters()}
    ok = all(g is not None and torch.isfinite(g).all() and g.abs().max() > 0
             for g in grads.values())
    check("all of U/V/gamma receive finite non-zero gradient at init", ok,
          ", ".join(f"|d{n}|max={g.abs().max():.2e}" for n, g in grads.items()))


def test_parameter_count(D=4096, L=32, r=32):
    lie = 2 * L * (2 * D * r + r)
    dense = (2 * L + 1) * D * D
    check("parameter count is O(L D r), not O(L D^2)", lie < dense / 50,
          f"lie={lie / 1e6:.1f}M vs respinquant dense={dense / 1e6:.1f}M "
          f"({dense / lie:.0f}x fewer)")


def main():
    print("low-rank Cayley residual transitions\n")
    print("[1] Woodbury form")
    test_woodbury_matches_dense()
    print("[2] exact orthogonality / SO(D)")
    test_orthogonal_and_special()
    print("[3] efficient application")
    test_apply_right()
    test_identity_at_zero_gate()
    print("[4] basis chain")
    test_basis_chain()
    print("[5] trainability")
    test_gradients()
    test_parameter_count()
    print("\nRESULT:", "PASS" if not FAILED else f"FAIL ({', '.join(FAILED)})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
