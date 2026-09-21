"""GPTQ x MXFP4 checks (needs a CUDA device: GPTQ.fasterquant synchronizes).

    python tests/test_mxfp4_gptq.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_utils.gptq_utils import GPTQ  # noqa: E402
from utils import quant_utils  # noqa: E402
from utils.mxfp4 import quantize_mx_fp4  # noqa: E402

DEV = "cuda"
BLOCK = 32
FP4_LEVELS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6.0])


def make_gptq(W, X, mxfp4=True):
    lin = nn.Linear(W.shape[1], W.shape[0], bias=False).to(DEV)
    lin.weight.data = W.clone().to(DEV)
    g = GPTQ(lin)
    g.add_batch(X.to(DEV), None)
    g.quantizer = quant_utils.WeightQuantizer()
    g.quantizer.configure(4, perchannel=True, sym=True, mxfp4=mxfp4, mx_block=BLOCK)
    return g


def on_mx_grid(Q):
    """Each 1x32 block of Q is (power-of-two scale) * E2M1: re-quantizing is a no-op."""
    return torch.equal(quantize_mx_fp4(Q, block=BLOCK, axis=-1), Q)


def proxy_loss(W, Q, H):
    D = (W - Q).double()
    return torch.trace(D @ H.double() @ D.T).item()


def block_hadamard(n, g):
    from scipy.linalg import hadamard

    Hg = torch.tensor(hadamard(g), dtype=torch.float32) / g**0.5
    return torch.block_diag(*[Hg] * (n // g))


def run(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond


def main():
    torch.manual_seed(0)
    ok = True
    out_f, in_f, n = 96, 256, 2048
    W = torch.randn(out_f, in_f) * 0.05
    W[:, 7] *= 30  # outlier input channel
    # correlated activations so the Hessian has real off-diagonal structure
    X = torch.randn(n, in_f) @ (torch.randn(in_f, in_f) * 0.2 + torch.eye(in_f))
    Hess = None

    # 1) diagonal Hessian => no cross-column compensation => GPTQ must equal RTN exactly
    g = make_gptq(W, torch.randn(n, in_f))
    g.H = torch.diag(torch.rand(in_f) + 0.5).to(DEV)
    g.fasterquant(percdamp=0.0)
    ok &= run("diag-H GPTQ == RTN quantize_mx_fp4",
              torch.equal(g.layer.weight.data.cpu(), quantize_mx_fp4(W, block=BLOCK, axis=-1)))

    # 2) real Hessian: on the MXFP4 grid, shape/dtype preserved, beats RTN on the proxy loss
    g = make_gptq(W, X)
    Hess = g.H.clone().cpu()
    g.fasterquant(percdamp=0.01)
    Q = g.layer.weight.data.cpu()
    rtn = quantize_mx_fp4(W, block=BLOCK, axis=-1)
    ok &= run("GPTQ output shape/dtype preserved", Q.shape == W.shape and Q.dtype == W.dtype)
    ok &= run("GPTQ output is on the MXFP4 grid", on_mx_grid(Q))
    ok &= run("GPTQ proxy loss < RTN proxy loss "
              f"({proxy_loss(W, Q, Hess):.4g} < {proxy_loss(W, rtn, Hess):.4g})",
              proxy_loss(W, Q, Hess) < proxy_loss(W, rtn, Hess))

    # 3) act-order: blocks must stay contiguous in the ORIGINAL column layout
    g = make_gptq(W, X)
    g.fasterquant(percdamp=0.01, actorder=True)
    Qa = g.layer.weight.data.cpu()
    ok &= run("act-order GPTQ output is on the MXFP4 grid (original layout)", on_mx_grid(Qa))
    ok &= run("act-order GPTQ proxy loss < RTN "
              f"({proxy_loss(W, Qa, Hess):.4g})", proxy_loss(W, Qa, Hess) < proxy_loss(W, rtn, Hess))

    # 4) rotation compatibility: fuse a block Hadamard (R on the input dim) first, then GPTQ
    for gsz in (32, 128):
        R = block_hadamard(in_f, gsz)
        Wr, Xr = W @ R, X @ R  # y = W X^T == (W R)(X R)^T
        g = make_gptq(Wr, Xr)
        Hr = g.H.clone().cpu()
        g.fasterquant(percdamp=0.01)
        Qr = g.layer.weight.data.cpu()
        ok &= run(f"hadamard(g={gsz})-rotated weight: layout kept, on grid",
                  Qr.shape == Wr.shape and on_mx_grid(Qr))
        ok &= run(f"hadamard(g={gsz})-rotated: GPTQ < RTN proxy loss",
                  proxy_loss(Wr, Qr, Hr) < proxy_loss(Wr, quantize_mx_fp4(Wr, block=BLOCK), Hr))

    # 5) layout guards
    g = make_gptq(W, X)
    try:
        g.fasterquant(blocksize=48)
        ok &= run("blocksize not multiple of mx_block is rejected", False)
    except AssertionError:
        ok &= run("blocksize not multiple of mx_block is rejected", True)

    # 6) tail block (in_features % 32 != 0) still works
    Wt = torch.randn(16, 80) * 0.1
    g = make_gptq(Wt, torch.randn(512, 80))
    g.fasterquant(percdamp=0.01)
    ok &= run("partial tail block: output equals block-wise RTN grid",
              on_mx_grid(g.layer.weight.data.cpu()))

    # 7) activation quantizer: MXFP4 forward is straight-through differentiable
    aq = quant_utils.ActQuantizer()
    aq.configure(4, sym=True, mxfp4=True, mx_block=BLOCK)
    x = torch.randn(4, 64, requires_grad=True)
    aq(x).sum().backward()
    ok &= run("activation MXFP4 has straight-through gradient", torch.equal(x.grad, torch.ones_like(x)))

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
