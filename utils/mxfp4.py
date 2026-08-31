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


def quantize_mx_fp4(x, block=32, axis=-1):
    """Return the MXFP4 fake-quantized version of ``x``.

    A shared E8M0 scale is applied per ``block`` elements along ``axis``. If the
    axis length is not a multiple of ``block`` it is padded (the padding never
    affects real elements: it only shares a block's scale, and is dropped on the
    way out).
    """
    if x.numel() == 0:
        return x
    x = x.movedim(axis, -1)
    orig_shape = x.shape
    H = orig_shape[-1]
    out_dtype = x.dtype
    xf = x.float()

    pad = (block - H % block) % block
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xf = xf.reshape(*xf.shape[:-1], xf.shape[-1] // block, block)  # [..., nblk, block]

    amax = xf.abs().amax(dim=-1, keepdim=True)                     # [..., nblk, 1]
    # E8M0 shared exponent (power-of-two scale).
    exp = torch.floor(torch.log2(amax.clamp(min=1e-30))) - _FP4_EMAX
    exp = torch.clamp(exp, -127, 127)
    scale = torch.pow(torch.tensor(2.0, device=xf.device), exp)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)

    scaled = xf / scale
    q = _round_to_fp4_mag(scaled.abs()) * torch.sign(xf)
    q = q * scale

    q = q.reshape(*orig_shape[:-1], -1)[..., :H]
    q = q.movedim(-1, axis)
    return q.to(out_dtype)
