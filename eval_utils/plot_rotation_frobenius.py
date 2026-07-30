# coding=utf-8
"""
각 레이어의 학습된 회전 행렬 (layers.{idx}.R1 / layers.{idx}.R2) 이
같은 크기의 Walsh-Hadamard 행렬과 얼마나 다른지를
레이어 인덱스(x축) vs Frobenius norm(y축) 으로 그린다.

배경
----
이 repo 의 random_hadamard_matrix() 는  matmul_hadU(diag(±1)) 로,
"Walsh-Hadamard 변환 + 열(column)별 랜덤 부호" 이다. 즉 R 의 초기값은
    R_init[:, j] = s_j * H[:, j]        (H = 정규화된 Walsh-Hadamard, s_j = ±1)
따라서 R 을 부호 없는 H 와 그냥 빼면 열마다 부호가 반대라 차이가 크게 나온다.
논문의 "R 이 (부호 제외) 스케일된 Hadamard 이다" 를 재려면 열별 부호를 맞춘 뒤
차이를 봐야 한다. 그래서 두 가지를 모두 그린다.

  - aligned : 열별 최적 부호 s_j = sign(<R[:,j], H[:,j]>) 로 맞춘 후 ||R - H_signed||_F
              (0 에 가까울수록 R 이 (부호 제외) Walsh-Hadamard 와 동일)
  - raw     : 부호 정렬 없이 ||R - H||_F  (참고용)

참조 H 는 repo 자신의 hadamard_matrix() (R 초기화와 동일한 변환) 로 생성해
완전히 일관된 비교를 한다.

사용 예
-------
  python eval_utils/plot_rotation_frobenius.py \
      --rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R.bin \
      --which both \
      --out figures/rotation_frobenius.png
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

# repo 루트를 path 에 추가 (utils.hadamard_utils import 위해)
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


def as_matrix(t):
    if isinstance(t, dict) and "weight" in t:
        t = t["weight"]
    return t.detach().to(torch.float64).cpu()


def discover_layers(state, which):
    """R.bin 키에서 layers.{idx}.{which} 형태의 (idx -> key) 매핑을 만든다."""
    pat = re.compile(rf"layers\.(\d+)\.{which}$")
    found = {}
    for k in state.keys():
        m = pat.search(k)
        if m:
            found[int(m.group(1))] = k
    return dict(sorted(found.items()))


_HAD_CACHE = {}


def get_reference_hadamard(size):
    """repo 와 동일한 정규화된 Walsh-Hadamard 행렬 (float64, CPU)."""
    if size not in _HAD_CACHE:
        _HAD_CACHE[size] = hadamard_matrix(size, "cpu").to(torch.float64).cpu()
    return _HAD_CACHE[size]


def frobenius_diffs(R, H):
    """(aligned, raw) Frobenius 차이를 반환.
    R, H : (D, D) torch.float64, 둘 다 정규화(직교)되어 있다고 가정.

    이 repo 의 random_hadamard_matrix() = matmul_hadU(diag(s)) = diag(s) @ H,
    즉 부호 s 는 '행(row)' 단위로 붙는다. 따라서 행별 최적 부호를 맞춘 뒤 비교한다:
        s_i = sign(<R[i,:], H[i,:]>),   H_signed = diag(s) @ H
    """
    # raw
    raw = torch.linalg.norm(R - H).item()
    # 행별 부호 정렬
    dots = (R * H).sum(dim=1)                 # (D,)  각 행의 내적
    s = torch.sign(dots)
    s[s == 0] = 1.0
    H_signed = H * s.unsqueeze(1)             # 행 i 에 s_i 곱
    aligned = torch.linalg.norm(R - H_signed).item()
    return aligned, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation_path", required=True, help="학습된 R.bin 경로")
    ap.add_argument("--which", choices=["R1", "R2", "both"], default="both",
                    help="비교할 행렬 종류")
    ap.add_argument("--out", default=None, help="저장 경로(png). 미지정 시 figures/ 아래 자동")
    ap.add_argument("--raw", action="store_true",
                    help="부호 미정렬 raw Frobenius 도 함께 그린다")
    args = ap.parse_args()

    state = torch.load(args.rotation_path, map_location="cpu")
    print(f"[load] {args.rotation_path}  (총 {len(state)} keys)")

    kinds = ["R1", "R2"] if args.which == "both" else [args.which]

    # 결과 수집
    results = {}   # kind -> (idxs, aligned[], raw[])
    for kind in kinds:
        layer_keys = discover_layers(state, kind)
        if not layer_keys:
            print(f"[warn] {kind} 키를 찾지 못했습니다. 건너뜀.")
            continue
        idxs, aligned_list, raw_list = [], [], []
        for idx, key in layer_keys.items():
            R = as_matrix(state[key])
            D = R.shape[0]
            H = get_reference_hadamard(D)
            aligned, raw = frobenius_diffs(R, H)
            idxs.append(idx)
            aligned_list.append(aligned)
            raw_list.append(raw)
        results[kind] = (idxs, aligned_list, raw_list)
        print(f"\n[{kind}]  (D={D}, {len(idxs)} layers)")
        print(f"  layer :  aligned ||R-H_signed||_F   |   raw ||R-H||_F")
        for i, a, r in zip(idxs, aligned_list, raw_list):
            print(f"  {i:>4d} :      {a:12.6f}          |   {r:12.6f}")
        print(f"  aligned  min/mean/max : {min(aligned_list):.4f} / "
              f"{np.mean(aligned_list):.4f} / {max(aligned_list):.4f}")

    if not results:
        raise SystemExit("R1/R2 키를 하나도 찾지 못했습니다.")

    # ---------------- 그림 ----------------
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"R1": "#1f77b4", "R2": "#d62728"}
    for kind, (idxs, aligned_list, raw_list) in results.items():
        c = colors.get(kind, None)
        ax.plot(idxs, aligned_list, marker="o", ms=4, lw=1.8, color=c,
                label=f"{kind}  (sign-aligned)")
        if args.raw:
            ax.plot(idxs, raw_list, marker="x", ms=4, lw=1.0, ls="--", color=c,
                    alpha=0.6, label=f"{kind}  (raw)")

    ax.set_xlabel("Layer index")
    ax.set_ylabel(r"$\|R - H_{\mathrm{Walsh}}\|_F$")
    ax.set_title("Rotation matrix vs. Walsh-Hadamard  (per-layer Frobenius distance)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = args.out
    if out is None:
        os.makedirs("figures", exist_ok=True)
        out = os.path.join("figures", "rotation_frobenius.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
