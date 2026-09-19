"""MXFP4 (OCP Microscaling FP4) fake-quantization.

Element format E2M1 (1 sign, 2 exponent, 1 mantissa) with the 16 representable
values +/- {0, 0.5, 1, 1.5, 2, 3, 4, 6}. A block of `block` consecutive elements
along one axis shares a single E8M0 (power-of-two) scale, following the OCP
Microscaling spec: the shared exponent is  floor(log2(block_amax)) - emax_elem,
with emax_elem = 2 (the exponent of the largest normal FP4 value, 6 = 1.5 * 2^2).

Only fake-quantization (quantize->dequantize back to the input dtype) is provided;
this is what the rotation / PTQ evaluation path needs.
"""

import torch

# E2M1 representable magnitudes (sign handled separately), ascending.
_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_MAX = 6.0
_FP4_EMAX = 2  # floor(log2(6.0))
FP4_SCALE = 3 / 4

_LEVEL_CACHE = {}
_MID_CACHE = {}


def _levels(device, dtype):
    key = (device, dtype)
    t = _LEVEL_CACHE.get(key)
    if t is None:
        t = torch.tensor(_FP4_LEVELS, device=device, dtype=dtype)
        _LEVEL_CACHE[key] = t
        _MID_CACHE[key] = (t[1:] + t[:-1]) / 2  # midpoints for round-to-nearest
    return t, _MID_CACHE[key]


def _round_to_fp4_mag(mag):
    """Round nonnegative magnitudes to the nearest E2M1 level (ties handled by
    bucketize's right=False; values above 6 clamp to 6)."""
    levels, mids = _levels(mag.device, mag.dtype)
    idx = torch.bucketize(mag, mids)
    return levels[idx]


_FP32_MIN_NORMAL = 2 ** (-127 + 1)


# ---- Scale casts (ported from DuQuant-v2/quantize/fp4_ops.py) ----------------
# Inputs are block abs-maxes / scales, so they are nonnegative by construction;
# the ``x.ge(0).all()`` assert of the original is dropped to avoid a GPU sync.
def cast_to_eBm0(x, ebits, emax):
    """Power-of-two (E8M0) scale: 2 ** (floor(log2(x)) - emax), clamped to the
    representable exponent range instead of overflowing."""
    assert ebits % 2 == 0, "EBm0 expects even number of bits"
    qmax = 2 ** (ebits - 1) - 1
    exponent = x.clamp(min=_FP32_MIN_NORMAL).log2().floor() - emax
    return 2 ** exponent.clamp(-qmax, qmax)


def cast_to_eBm0_improved(x):
    """Log-uniform 256-level scale grid spanning [min(x), max(x)] of the whole
    tensor (finer than power-of-two; rounds down on the log grid)."""
    log_min = x.min().clamp(min=_FP32_MIN_NORMAL).log2()
    log_max = x.max().clamp(min=_FP32_MIN_NORMAL).log2()
    span = (log_max - log_min).clamp(min=1e-12)  # span == 0 -> grid collapses to x.min()
    step = (255 * (x.clamp(min=_FP32_MIN_NORMAL).log2() - log_min) / span).floor().clamp(0, 255)
    return 2 ** (span * step / 255 + log_min)


def _mx_block_scale(amax, scale_mode):
    """Stage 1: per-block shared scale from the block abs-max ``amax`` [..., nblk, 1].

    ``e8m0``   : OCP MX spec, power-of-two scale (cast_to_eBm0).
    ``log256`` : finer log-uniform scale grid (cast_to_eBm0_improved) on amax / 6.
    All-zero blocks get scale 1 (their elements are 0 anyway).
    """
    nonzero = amax > 0
    if scale_mode == "e8m0":
        scale = cast_to_eBm0(amax, ebits=8, emax=_FP4_EMAX)
    elif scale_mode == "log256":
        raw = amax / _FP4_MAX
        # zero blocks would pin the grid minimum to FP32_MIN_NORMAL; keep them out of min/max
        raw = torch.where(nonzero, raw, raw.max())
        scale = cast_to_eBm0_improved(raw)
    else:
        raise ValueError(f"unknown scale_mode {scale_mode!r}")
    return torch.where(nonzero, scale, torch.ones_like(scale))


def _quantize_with_scale(xf, scale):
    """Stage 2: scale elements, round to the nearest E2M1 level, rescale."""
    q = _round_to_fp4_mag((xf / scale).abs()) * torch.sign(xf)
    return q * scale


def quantize_mx_fp4(x, block=32, axis=-1, scale_mode="e8m0"):
    """Return the MXFP4 fake-quantized version of ``x``.

    Two stages per block of ``block`` consecutive elements along ``axis``:
      1. shared scale from the block abs-max (``_mx_block_scale``; ``scale_mode``
         is ``"e8m0"`` for the OCP power-of-two scale or ``"log256"`` for the
         finer scale grid),
      2. element quantization to E2M1 with that scale (``_quantize_with_scale``).

    If the axis length is not a multiple of ``block`` it is padded (the padding
    never affects real elements: it only shares a block's scale, and is dropped
    on the way out).
    """
    if x.numel() == 0:
        return x
    x = x.movedim(axis, -1)
    orig_shape = x.shape # [Batch, Seq_len, hidden_dim]
    H = orig_shape[-1] # hidden_dim
    out_dtype = x.dtype
    xf = x.float() #

    pad = (block - H % block) % block # 
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xf = xf.reshape(*xf.shape[:-1], xf.shape[-1] // block, block)  # [..., nblk, block]

    amax = xf.abs().amax(dim=-1, keepdim=True)                     # [..., nblk, 1]
    scale = _mx_block_scale(amax, scale_mode)                      # stage 1
    q = _quantize_with_scale(xf, scale)                            # stage 2

    q = q.reshape(*orig_shape[:-1], -1)[..., :H]
    q = q.movedim(-1, axis)
    return q.to(out_dtype)
