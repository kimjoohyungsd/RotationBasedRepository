# coding=utf-8
"""
R.bin 안의 학습된 회전 행렬(model.layers.{idx}.R1 / model.layers.{idx}.R2)이
논문 그림처럼 "스케일된 Hadamard 행렬"이며, R1^T @ R2 ≈ I 인지를 확인/시각화한다.

핵심 아이디어
-------------
hidden_size = D 인 정규화된 (randomized) Hadamard 행렬의 원소는 ±1/sqrt(D) 이다.
D = 4096 이면 1/sqrt(D) = 1/64 = 2^-6 이므로, R1 * 2^6 의 원소는 ≈ ±1 이 되어
Hadamard 구조(±1 패턴)를 그대로 드러낸다. (스케일은 자동으로 sqrt(D) 로 계산)

R1 과 R2 가 (거의) 같은 회전이면 R1^T @ R2 ≈ I 가 성립한다.

사용 예
-------
  python eval_utils/plot_rotation_hadamard.py \
      --rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R.bin \
      --layer 0 \
      --block 4 \
      --out figures/rotation_check/layer0.png
"""

import argparse
import math
import os

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib 가 필요합니다. torch/matplotlib 가 모두 있는 env(예: flatquant)에서 실행하세요."
    ) from e


def find_key(state, idx, which):
    """R.bin 의 실제 키를 찾는다. 저장 방식에 따라 'model.' 접두어가 다를 수 있어
    여러 후보를 시도한다. which in {'R1', 'R2', 'head_R2'}."""
    if which == "R1":
        suffixes = [f"layers.{idx}.R1"]
    elif which == "R2":
        suffixes = [f"layers.{idx}.R2"]
    elif which == "head_R2":
        suffixes = [f"layers.{idx}.self_attn.R2"]
    else:
        raise ValueError(which)

    candidates = []
    for suf in suffixes:
        candidates += [
            f"model.{suf}",
            f"model.model.{suf}",
            suf,
        ]
    for c in candidates:
        if c in state:
            return c
    # 마지막 수단: 부분 일치 검색
    for k in state.keys():
        if any(k.endswith(suf) for suf in suffixes):
            return k
    raise KeyError(
        f"layer {idx} 의 {which} 키를 찾을 수 없습니다. "
        f"R.bin 에 존재하는 키 예시: {list(state.keys())[:6]} ..."
    )


def as_matrix(t):
    """RotateModule 가중치이든 순수 텐서이든 (D, D) float64 numpy 로 변환."""
    if isinstance(t, dict) and "weight" in t:
        t = t["weight"]
    return t.detach().to(torch.float64).cpu().numpy()


def draw_heatmap(ax, mat, title, scale_label=None, integer=False, vlim=None):
    """±값을 셀에 숫자로 찍는 히트맵. 파랑=양수, 빨강=음수 (논문 그림과 동일)."""
    n = mat.shape[0]
    if vlim is None:
        vlim = np.max(np.abs(mat)) or 1.0
    # RdBu: 높은 값 -> 파랑, 낮은 값 -> 빨강  (논문 그림의 blue=+1, red=-1 과 일치)
    ax.imshow(mat, cmap="RdBu", vmin=-vlim, vmax=vlim)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if integer:
                txt = f"{int(round(v)):d}"
            elif abs(v) < 1e-2 and abs(v) > 0:
                txt = f"{v:.0e}"
            else:
                txt = f"{v:.2f}"
            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=9,
                color="black",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    if scale_label:
        ax.set_xlabel(scale_label, fontsize=9)


def report_metrics(R1, R2, scale):
    """정량 검증 지표를 콘솔에 출력."""
    D = R1.shape[0]
    I = np.eye(D)

    # 1) Hadamard 일치도: R1*scale 의 원소가 ±1 에 얼마나 가까운지
    dev_R1 = np.abs(np.abs(R1 * scale) - 1.0)
    dev_R2 = np.abs(np.abs(R2 * scale) - 1.0)

    # 2) 직교성: R^T R = I 여야 함
    orth_R1 = np.max(np.abs(R1.T @ R1 - I))
    orth_R2 = np.max(np.abs(R2.T @ R2 - I))

    # 3) R1 ≈ R2  =>  R1^T @ R2 ≈ I
    RtR = R1.T @ R2
    resid = RtR - I
    fro = np.linalg.norm(resid, "fro")
    maxabs = np.max(np.abs(resid))

    print("\n================ 정량 검증 지표 ================")
    print(f"행렬 크기 D                 : {D}  (scale = sqrt(D) = {scale:.4f}"
          f"{' = 2^%d' % round(math.log2(scale)) if abs(scale - 2**round(math.log2(scale))) < 1e-6 else ''})")
    print(f"|R1*scale| 이 1 에서 벗어난 최대/평균 : {dev_R1.max():.4e} / {dev_R1.mean():.4e}")
    print(f"|R2*scale| 이 1 에서 벗어난 최대/평균 : {dev_R2.max():.4e} / {dev_R2.mean():.4e}")
    print(f"  -> 0 에 가까울수록 완벽한 (스케일된) Hadamard 구조")
    print(f"직교성 max|R1^T R1 - I|      : {orth_R1:.4e}")
    print(f"직교성 max|R2^T R2 - I|      : {orth_R2:.4e}")
    print(f"||R1^T R2 - I||_F           : {fro:.4e}")
    print(f"max|R1^T R2 - I|            : {maxabs:.4e}")
    print(f"  -> 0 에 가까울수록 R1 과 R2 가 동일한 회전 (R1^T R2 ≈ I)")
    print("===============================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotation_path", required=True, help="학습된 R.bin 경로")
    ap.add_argument("--layer", type=int, default=0, help="확인할 레이어 인덱스")
    ap.add_argument("--block", type=int, default=4, help="좌상단 시각화 블록 크기 (b x b)")
    ap.add_argument("--out", default=None, help="저장할 그림 경로(png). 미지정 시 figures/ 아래 자동")
    ap.add_argument("--scale", type=float, default=None,
                    help="R 에 곱할 스케일. 미지정 시 sqrt(D) 자동 (D=4096 -> 64=2^6)")
    args = ap.parse_args()

    state = torch.load(args.rotation_path, map_location="cpu")
    print(f"[load] {args.rotation_path}  (총 {len(state)} keys)")

    k1 = find_key(state, args.layer, "R1")
    k2 = find_key(state, args.layer, "R2")
    print(f"[key ] R1 = {k1}")
    print(f"[key ] R2 = {k2}")

    R1 = as_matrix(state[k1])
    R2 = as_matrix(state[k2])
    assert R1.shape == R2.shape and R1.shape[0] == R1.shape[1], \
        f"R1{R1.shape}, R2{R2.shape} 형태가 정사각/일치하지 않음"

    D = R1.shape[0]
    scale = args.scale if args.scale is not None else math.sqrt(D)
    exp = round(math.log2(scale))
    scale_str = f"$2^{{{exp}}}$" if abs(scale - 2 ** exp) < 1e-6 else f"{scale:.2f}"

    report_metrics(R1, R2, scale)

    b = args.block
    R1b = R1[:b, :b] * scale
    R2b = R2[:b, :b] * scale
    Ib = (R1.T @ R2)[:b, :b]
    Had_ref = np.sign(R1b)  # R1 이 실제로 Hadamard 라면 이 ±1 패턴을 따른다

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]})

    draw_heatmap(axes[0], Had_ref, "Hadamard matrix\n(sign of $2^{k}\\!\\cdot\\!R_1$)",
                 integer=True, vlim=1)
    draw_heatmap(axes[1], R1b, f"{scale_str} $\\times R_1$[:{b},:{b}]", vlim=1.2)
    draw_heatmap(axes[2], R2b, f"{scale_str} $\\times R_2$[:{b},:{b}]", vlim=1.2)
    draw_heatmap(axes[3], Ib, f"$(R_1^T\\!@\\,R_2)$[:{b},:{b}]", vlim=1)
    draw_heatmap(axes[4], np.eye(b), "I", integer=True, vlim=1)

    # 화살표(패널3->4)와 ≈ 기호(패널4~5) 를 figure 좌표에 추가
    fig.text(0.615, 0.5, r"$\Rightarrow$", fontsize=22, ha="center", va="center")
    fig.text(0.805, 0.5, r"$\approx$", fontsize=22, ha="center", va="center")

    fig.suptitle(
        f"Layer {args.layer}:  {k1}  &  {k2}", fontsize=12, y=1.02
    )
    fig.tight_layout()

    out = args.out
    if out is None:
        os.makedirs("figures", exist_ok=True)
        out = os.path.join("figures", f"rotation_check_layer{args.layer}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
