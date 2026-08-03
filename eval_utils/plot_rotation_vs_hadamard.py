# coding=utf-8
"""학습된 회전 행렬이 Walsh-Hadamard 에서 얼마나 멀어졌는지를 레이어별로 그린다.

한 그림에 세 개의 선을 그린다 (x = decoder layer index):

  R1            layers.{i}.R1            Attention 블록의 입력 basis
  R2            layers.{i}.R2            FFN 블록의 입력 basis (= attention 출력 basis)
  self_attn.R2  layers.{i}.self_attn.R2  v_proj 와 o_proj 사이의 head 회전 (논문의 R3)

두 가지 보정을 기본으로 넣는다. 둘 다 끄면 그림이 오해를 부른다.

1) 크기 정규화 (--absolute 로 해제)
   R1/R2 는 D=hidden_size (Llama-2-7B 는 4096), self_attn.R2 는 D=head_dim (128) 이다.
   직교행렬은 ||H||_F = sqrt(D) 이므로 ||R-H||_F 자체가 sqrt(D) 에 비례해서 커진다.
   그대로 겹쳐 그리면 4096 차원 선이 항상 위에 오는 착시가 생긴다. 그래서 기본은
       ||R - H||_F / sqrt(D)
   로 나눠 [0, 2] 범위의 무차원 값으로 만든다 (0 = H 와 동일, sqrt(2) ~ 무상관, 2 = -H).

2) 부호 정렬 (--raw 로 unaligned 값도 함께 표시)
   optimize_rotation.py 에서 R1/R2 는 hadamard_matrix() (부호 없는 Walsh-Hadamard) 로,
   self_attn.R2 는 random_hadamard_matrix() = diag(±1) @ H 로 초기화된다. 즉 R3 는
   시작부터 행 부호가 랜덤이라 부호를 맞추지 않고 H 와 빼면 "학습으로 멀어진 정도"가
   아니라 "초기 랜덤 부호"를 재게 된다. 그래서 행별 최적 부호
       s_i = sign(<R[i,:], H[i,:]>),  H_signed = diag(s) @ H
   로 맞춘 뒤의 거리를 기본값으로 쓴다.

사용 예
-------
  python eval_utils/plot_rotation_vs_hadamard.py \
      --rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B-hf/ReSpinQuant/R.bin \
      --out figures/respinquant_llama2_7b_rotation_vs_hadamard.png

  # 정규화 없이 원래 Frobenius 값 + 부호 미정렬 값까지
  python eval_utils/plot_rotation_vs_hadamard.py --rotation_path ... --absolute --raw

  # 두 체크포인트 비교 (ReSpinQuant vs LieReSpinQuant)
  python eval_utils/plot_rotation_vs_hadamard.py \
      --rotation_path A/R.bin B/R.bin --labels ReSpinQuant LieReSpinQuant
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.hadamard_utils import hadamard_matrix  # noqa: E402

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib 가 필요합니다. torch/matplotlib 가 모두 있는 env(예: flatquant)에서 실행하세요."
    ) from e


# 표시 이름 -> R.bin 키 패턴. self_attn.R2 를 먼저 확인해야 layers.{i}.R2 와 안 섞인다.
SERIES = [
    ("R1", r"layers\.(\d+)\.R1$", "Attention input basis", "#1f77b4", "o"),
    ("R2", r"layers\.(\d+)\.R2$", "FFN input basis", "#d62728", "s"),
    ("self_attn.R2", r"layers\.(\d+)\.self_attn\.R2$",
     "Head rotation (v_proj -> o_proj)", "#2ca02c", "^"),
]

_HAD_CACHE = {}


def reference_hadamard(size):
    """repo 자신의 hadamard_matrix() 로 만든 정규화된 Walsh-Hadamard (float64, CPU)."""
    if size not in _HAD_CACHE:
        _HAD_CACHE[size] = hadamard_matrix(size, "cpu").to(torch.float64).cpu()
    return _HAD_CACHE[size]


def as_matrix(t):
    if isinstance(t, dict) and "weight" in t:
        t = t["weight"]
    return t.detach().to(torch.float64).cpu()


def discover(state, pattern):
    pat = re.compile(pattern)
    found = {}
    for k in state.keys():
        if not isinstance(k, str):
            continue
        m = pat.search(k)
        if m:
            found[int(m.group(1))] = k
    return dict(sorted(found.items()))


def frobenius(R, H):
    """(aligned, raw) 거리. 둘 다 정규화된 직교행렬이라고 가정."""
    raw = torch.linalg.norm(R - H).item()
    dots = (R * H).sum(dim=1)          # 행별 내적
    s = torch.sign(dots)
    s[s == 0] = 1.0
    aligned = torch.linalg.norm(R - H * s.unsqueeze(1)).item()
    return aligned, raw


def collect(path, normalize):
    """한 R.bin 에서 세 계열의 (layer -> distance) 를 뽑는다."""
    try:
        state = torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        state = torch.load(path, map_location="cpu")
    print(f"[load] {path}  ({len(state)} keys)")

    out = {}
    for name, pattern, _desc, _c, _m in SERIES:
        keys = discover(state, pattern)
        if not keys:
            print(f"  [warn] '{name}' 키 없음 -> 건너뜀")
            continue
        idxs, aligned, raw, dim = [], [], [], None
        for idx, key in keys.items():
            R = as_matrix(state[key])
            dim = R.shape[0]
            a, r = frobenius(R, reference_hadamard(dim))
            if normalize:
                scale = float(np.sqrt(dim))
                a, r = a / scale, r / scale
            idxs.append(idx)
            aligned.append(a)
            raw.append(r)
        out[name] = dict(idx=idxs, aligned=aligned, raw=raw, dim=dim)
        print(f"  {name:<13} D={dim:<5} layers={len(idxs):<3} "
              f"aligned min/mean/max = {min(aligned):.4f} / "
              f"{np.mean(aligned):.4f} / {max(aligned):.4f}")
    if not out:
        raise SystemExit(f"{path}: R1 / R2 / self_attn.R2 키를 하나도 찾지 못했습니다.")
    return out


def print_table(label, data, normalize):
    unit = "normalized" if normalize else "absolute"
    print(f"\n=== {label}  ({unit} ||R - H_signed||_F) ===")
    names = [n for n, *_ in SERIES if n in data]
    print("layer | " + " | ".join(f"{n:>13}" for n in names))
    n_layers = max(len(data[n]["idx"]) for n in names)
    for i in range(n_layers):
        row = f"{i:>5} |"
        for n in names:
            d = data[n]
            row += f" {d['aligned'][i]:>13.6f} |" if i < len(d["idx"]) else f" {'-':>13} |"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation_path", nargs="+", required=True,
                    help="학습된 R.bin 경로 (여러 개 주면 겹쳐 그린다)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="각 R.bin 의 범례 이름 (미지정 시 상위 디렉토리명)")
    ap.add_argument("--out", default=None, help="저장 경로(png)")
    ap.add_argument("--absolute", action="store_true",
                    help="sqrt(D) 정규화를 끄고 원래 Frobenius 값을 쓴다 "
                         "(R1/R2 와 self_attn.R2 는 차원이 달라 직접 비교 불가)")
    ap.add_argument("--raw", action="store_true",
                    help="행별 부호 정렬을 하지 않은 거리도 점선으로 함께 그린다")
    ap.add_argument("--logy", action="store_true",
                    help="y축 로그 스케일. 값이 보통 1e-4 ~ 1e-2 로 두 자릿수 넘게 "
                         "퍼져 있어서 레이어별 구조를 보려면 대개 이쪽이 낫다.")
    ap.add_argument("--ref_line", action="store_true",
                    help="sqrt(2)(= H 와 무상관) 기준선을 그린다. 학습된 회전은 "
                         "그보다 훨씬 작아서 켜면 데이터가 바닥에 눌린다.")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    normalize = not args.absolute
    labels = args.labels or [
        os.path.basename(os.path.dirname(p.rstrip("/"))) or f"ckpt{i}"
        for i, p in enumerate(args.rotation_path)
    ]
    if len(labels) != len(args.rotation_path):
        raise SystemExit("--labels 개수가 --rotation_path 개수와 다릅니다.")

    runs = []
    for path, label in zip(args.rotation_path, labels):
        data = collect(path, normalize)
        print_table(label, data, normalize)
        runs.append((label, data))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    linestyles = ["-", "--", ":", "-."]
    for run_i, (label, data) in enumerate(runs):
        ls = linestyles[run_i % len(linestyles)]
        for name, _pattern, desc, color, marker in SERIES:
            if name not in data:
                continue
            d = data[name]
            suffix = f" [{label}]" if len(runs) > 1 else ""
            ax.plot(d["idx"], d["aligned"], color=color, marker=marker, ms=4,
                    lw=1.8, ls=ls,
                    label=f"{name} - {desc} (D={d['dim']}){suffix}")
            if args.raw:
                ax.plot(d["idx"], d["raw"], color=color, marker="x", ms=3,
                        lw=1.0, ls="--", alpha=0.45,
                        label=f"{name} - sign-unaligned{suffix}")

    ax.set_xlabel("Decoder layer index")
    if normalize:
        ax.set_ylabel(r"$\|R - H_{\mathrm{Walsh}}\|_F \,/\, \sqrt{D}$")
        if args.ref_line:
            # 학습된 회전은 보통 H 근처(~1e-3)에 머물러서, 이 선을 그리면 축이
            # sqrt(2) 까지 늘어나 실제 변화가 바닥에 눌린다. 기본은 끈다.
            ax.axhline(np.sqrt(2), color="gray", lw=0.8, ls=":", alpha=0.7)
            ax.text(0.995, np.sqrt(2), r" $\sqrt{2}$ = uncorrelated", color="gray",
                    fontsize=8, va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())
    else:
        ax.set_ylabel(r"$\|R - H_{\mathrm{Walsh}}\|_F$")
    if args.logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(args.title or
                 "Learned rotations vs. Walsh-Hadamard, per decoder layer"
                 + ("" if normalize else "  (absolute -- dimensions differ!)"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    out = args.out
    if out is None:
        os.makedirs("figures", exist_ok=True)
        out = os.path.join("figures", "rotation_vs_hadamard.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
