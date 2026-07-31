# coding=utf-8
"""Low-rank Lie-algebra (Cayley) parameterization of residual basis transitions.

Motivation
----------
ReSpinQuant learns one dense orthogonal basis per residual boundary
(``layers[i].R1``, ``layers[i].R2``, ``R1_final`` -- 2L+1 matrices of D x D, i.e.
~1.09e9 parameters for LLaMA-3 8B) and only afterwards compresses each residual
transition ``T = R_in^T R_out`` with a truncated SVD.  That has three problems:

1. the SVD minimises an *unweighted* Frobenius error on ``T - I``, which is not
   the quantities we care about (activation-weighted / output error);
2. ``U_sub V_sub^T`` lands in O(r) but not necessarily SO(r) without an explicit
   determinant fix;
3. the rank is chosen post-hoc and uniformly across layers.

This module removes the post-hoc step entirely: the *transition itself* is the
learned object, and it is low-rank and exactly orthogonal by construction.

Parameterization
----------------
For each residual boundary k, learn ``U_k, V_k in R^{D x r}`` and gates
``gamma_k in R^r`` and form the skew-symmetric generator

    A_k = U_k diag(gamma_k) V_k^T - V_k diag(gamma_k) U_k^T ,   A_k^T = -A_k

which factors as ``A_k = P_k M_k P_k^T`` with

    P_k = [U_k, V_k] in R^{D x 2r},
    M_k = [[0, G_k], [-G_k, 0]] in R^{2r x 2r},   G_k = diag(gamma_k).

The transition is the Cayley transform

    dR_k = (I - A_k)^{-1} (I + A_k) .

Because ``A_k`` is skew, ``dR_k`` is orthogonal *exactly* and always has
determinant +1, i.e. it is in SO(D) with no determinant correction needed.
Because ``A_k`` has rank <= 2r, the Woodbury identity collapses the Cayley
transform to a rank-2r update

    dR_k = I + P_k Z_k P_k^T ,   Z_k = 2 (I_{2r} - M_k P_k^T P_k)^{-1} M_k ,

so materialising it costs O(D r^2 + r^3) and *applying* it costs O(D r) per
token instead of O(D^2).  See :func:`cayley_lowrank_factors`.

Basis chain
-----------
Bases are defined recursively from a fixed Hadamard base ``B_0``:

    B_{k+1} = B_k dR_k          =>      B_k^T B_{k+1} = dR_k

so the transition that the residual stream actually has to pay for at inference
time *is* the learned low-rank rotation -- nothing is approximated afterwards.
With ``gamma = 0`` every ``dR_k = I`` and the whole chain degenerates to a single
global rotation, i.e. plain SpinQuant; the gates are initialised near (but not
at) zero so that training starts from that point with non-zero gradients.

Adaptive rank
-------------
``gamma_k`` acts as a per-direction gate.  An L1 penalty on the gates
(:func:`gate_l1`) drives unused directions to exactly the identity, which lets
each boundary settle on its own effective rank instead of a uniform r.
:meth:`LowRankCayley.effective_rank` reports how many gates survive.
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn


def _skew_block(gamma: torch.Tensor) -> torch.Tensor:
    """M = [[0, G], [-G, 0]] for G = diag(gamma), shape (2r, 2r)."""
    r = gamma.shape[0]
    M = gamma.new_zeros(2 * r, 2 * r)
    idx = torch.arange(r, device=gamma.device)
    M[idx, r + idx] = gamma
    M[r + idx, idx] = -gamma
    return M


def cayley_lowrank_factors(
    U: torch.Tensor, V: torch.Tensor, gamma: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factors (P, Z) with ``Cayley(A) = I + P Z P^T`` for the low-rank skew

        A = U diag(gamma) V^T - V diag(gamma) U^T .

    Derivation: write ``A = P M P^T``.  Woodbury gives
    ``(I - P M P^T)^{-1} = I + P (I - M P^T P)^{-1} M P^T`` and therefore
    ``Cayley(A) = I + 2 (I - A)^{-1} A = I + 2 P (I - M P^T P)^{-1} M P^T``.

    Args:
        U, V: ``(D, r)``
        gamma: ``(r,)``

    Returns:
        P: ``(D, 2r)``, Z: ``(2r, 2r)``
    """
    assert U.shape == V.shape, (U.shape, V.shape)
    assert gamma.shape[0] == U.shape[1], (gamma.shape, U.shape)
    P = torch.cat([U, V], dim=1)                       # D x 2r
    M = _skew_block(gamma)                             # 2r x 2r
    two_r = M.shape[0]
    eye = torch.eye(two_r, dtype=P.dtype, device=P.device)
    Z = 2.0 * torch.linalg.solve(eye - M @ (P.transpose(0, 1) @ P), M)
    return P, Z


def cayley_dense(A: torch.Tensor) -> torch.Tensor:
    """Reference (slow) Cayley transform ``(I - A)^{-1}(I + A)`` for skew A."""
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(eye - A, eye + A)


class LowRankCayley(nn.Module):
    """One residual basis transition ``dR = Cayley(A)`` with ``rank(A) <= 2r``.

    Args:
        dim: hidden size D.
        rank: r, the number of learned rotation planes (the generator has rank
            at most 2r, i.e. up to 2r non-trivial rotation directions).
        gate_init: std of the initial gates.  0 gives exactly ``dR = I`` but also
            zero gradient w.r.t. U/V, so a small non-zero value is the default.
    """

    def __init__(self, dim: int, rank: int, gate_init: float = 1e-2,
                 dtype: torch.dtype = torch.float32, device=None):
        super().__init__()
        assert 0 < rank <= dim // 2, f"rank must be in (0, {dim // 2}], got {rank}"
        self.dim = dim
        self.rank = rank
        scale = 1.0 / math.sqrt(dim)
        self.U = nn.Parameter(torch.randn(dim, rank, dtype=dtype, device=device) * scale)
        self.V = nn.Parameter(torch.randn(dim, rank, dtype=dtype, device=device) * scale)
        self.gamma = nn.Parameter(
            torch.randn(rank, dtype=dtype, device=device) * gate_init)

    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(P, Z) such that ``dR = I + P Z P^T``."""
        return cayley_lowrank_factors(self.U, self.V, self.gamma)

    def generator(self) -> torch.Tensor:
        """The dense skew generator A (D x D).  For testing / inspection only."""
        G = torch.diag(self.gamma)
        return self.U @ G @ self.V.t() - self.V @ G @ self.U.t()

    def matrix(self) -> torch.Tensor:
        """The dense transition ``dR`` (D x D).  Needed only to fuse bases offline."""
        P, Z = self.factors()
        eye = torch.eye(self.dim, dtype=P.dtype, device=P.device)
        return eye + P @ Z @ P.transpose(0, 1)

    def apply_right(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ dR`` in O(D r) per row, without materialising dR."""
        P, Z = self.factors()
        return x + ((x @ P) @ Z) @ P.transpose(0, 1)

    @torch.no_grad()
    def effective_rank(self, tol: float = 1e-4) -> int:
        """Number of gates that are still meaningfully open."""
        return int((self.gamma.abs() > tol).sum().item())

    def gate_l1(self) -> torch.Tensor:
        return self.gamma.abs().sum()

    def extra_repr(self) -> str:
        return f"dim={self.dim}, rank={self.rank}"


class LieBasisChain(nn.Module):
    """The 2L+1 residual-stream bases of a decoder stack, defined recursively.

    ``bases()[k]`` is the orthogonal basis at boundary k, with

        bases[0]   = B0                      (fixed Hadamard)
        bases[k+1] = bases[k] @ deltas[k]

    Boundary indexing matches ReSpinQuant's naming:

        bases[2i]   = layers[i].R1   (attention input / previous FFN output)
        bases[2i+1] = layers[i].R2   (attention output / FFN input)
        bases[2L]   = R1_final       (last FFN output, un-rotated before lm_head)

    so ``deltas[2i]`` is the attention-skip transition of layer i and
    ``deltas[2i+1]`` its FFN-skip transition.
    """

    def __init__(self, dim: int, num_layers: int, rank: int, base: torch.Tensor,
                 gate_init: float = 1e-2, device=None):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.rank = rank
        self.num_transitions = 2 * num_layers
        self.register_buffer("base", base.to(torch.float32), persistent=True)
        self.deltas = nn.ModuleList([
            LowRankCayley(dim, rank, gate_init=gate_init, device=device)
            for _ in range(self.num_transitions)
        ])

    def transition_factors(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """(P, Z) for every transition, in boundary order."""
        return [d.factors() for d in self.deltas]

    def bases(self, factors: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None
              ) -> List[torch.Tensor]:
        """Materialise all 2L+1 bases by accumulating the low-rank updates.

        Each step is ``B_{k+1} = B_k + (B_k P_k) Z_k P_k^T`` -- two D x 2r
        products rather than a D x D x D matmul.
        """
        if factors is None:
            factors = self.transition_factors()
        out = [self.base]
        cur = self.base
        for (P, Z) in factors:
            cur = cur + ((cur @ P) @ Z) @ P.transpose(0, 1)
            out.append(cur)
        return out

    def gate_l1(self) -> torch.Tensor:
        return sum(d.gate_l1() for d in self.deltas)

    @torch.no_grad()
    def effective_ranks(self, tol: float = 1e-4) -> List[int]:
        return [d.effective_rank(tol) for d in self.deltas]

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for d in self.deltas:
            params += [d.U, d.V, d.gamma]
        return params

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, num_layers={self.num_layers}, rank={self.rank}, "
                f"transitions={self.num_transitions}")


def state_dict_to_factors(state: dict, prefix: str, num_transitions: int,
                          device="cpu", dtype=torch.float32
                          ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Rebuild the (P, Z) factors of every transition from a saved checkpoint."""
    factors = []
    for k in range(num_transitions):
        U = state[f"{prefix}deltas.{k}.U"].to(device=device, dtype=dtype)
        V = state[f"{prefix}deltas.{k}.V"].to(device=device, dtype=dtype)
        g = state[f"{prefix}deltas.{k}.gamma"].to(device=device, dtype=dtype)
        factors.append(cayley_lowrank_factors(U, V, g))
    return factors


def bases_from_factors(base: torch.Tensor,
                       factors: Sequence[Tuple[torch.Tensor, torch.Tensor]]
                       ) -> List[torch.Tensor]:
    """Same accumulation as :meth:`LieBasisChain.bases`, for offline fusion."""
    out = [base]
    cur = base
    for (P, Z) in factors:
        cur = cur + ((cur @ P) @ Z) @ P.transpose(0, 1)
        out.append(cur)
    return out
