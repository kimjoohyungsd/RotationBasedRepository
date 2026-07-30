# coding=utf-8
"""학습된 Rotation Matrix가 초기값(Walsh-Hadamard)에서 얼마나 벗어났는지 측정 + 시각화.

optimize_rotation.py 가 저장한 R.bin 을 읽어서 각 rotation 을 초기값과 비교한다.

ReSpinQuant (--respinquant) 로 학습한 경우 R.bin 의 key 구성:
    model.layers.{i}.R1           (hidden, hidden)  init = hadamard_matrix        -> 결정론적 Walsh-Hadamard
    model.layers.{i}.R2           (hidden, hidden)  init = hadamard_matrix        -> 결정론적 Walsh-Hadamard
    model.R1_final                (hidden, hidden)  init = hadamard_matrix        -> 결정론적 Walsh-Hadamard
    model.layers.{i}.self_attn.R2 (head,   head)    init = random_hadamard_matrix -> diag(±1) @ H (부호는 랜덤)

SpinQuant (기존) 로 학습한 경우:
    R1                            (hidden, hidden)  init = random_hadamard_matrix
    model.layers.{i}.self_attn.R2 (head,   head)    init = random_hadamard_matrix

random_hadamard_matrix 는 `diag(s) @ hadamard_matrix(n)` 이므로 (matmul_hadU 가 행 단위로
Hadamard 를 적용한다), 학습 후에는 s 를 알 수 없다. 이런 행렬은 행별 부호를
s_i = sign(<R_i, H_i>) 로 복원한 뒤 비교한다(= sign-aligned). 결정론적 Walsh-Hadamard 로
초기화된 R1/R2/R1_final 은 부호 보정 없이 그대로 비교하면 된다.

측정 지표 (모두 float64):
    cos_matrix   = <R, H>_F / (||R||_F ||H||_F)   (행별 cosine 의 평균과 동일)
    row_cos      = 행 i 끼리의 cosine (min/mean/max 로 분포를 본다)
    angle_deg    = arccos(row_cos) 의 평균 (축이 평균 몇 도 돌아갔는지)
    frob         = ||R - H||_F
    rel_frob     = ||R - H||_F / ||H||_F          (직교행렬이면 = sqrt(2(1 - cos_matrix)))
    orth_err     = ||R^T R - I||_F / sqrt(n)      (Stiefel SGD 가 직교성을 유지했는지 sanity check)

사용 예:
    python analyze_rotation_drift.py \
        --rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R.bin \
        --output_dir figures/rotation_drift --tag llama2-7b
"""

import argparse
import csv
import os
import re
import sys

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# utils.hadamard_utils -> utils.utils -> fast_hadamard_transform (CUDA 확장) 순으로 import 가 걸려 있다.
# matmul_hadU 자체는 순수 PyTorch 라서 CPU 에서도 돌기 때문에, 확장이 없으면 stub 을 끼워 넣는다.
try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:  # pragma: no cover
    import types

    _stub = types.ModuleType("fast_hadamard_transform")
    _stub.hadamard_transform = None
    sys.modules["fast_hadamard_transform"] = _stub

from utils.hadamard_utils import hadamard_matrix  # noqa: E402


# ----------------------------------------------------------------------------- 색상 (light / dark)
PALETTE = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#d9d8d4",
        "R1": "#2a78d6",
        "R2": "#eb6834",
        "R3": "#1baf7a",
        "R1_final": "#4a3aa7",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#3a3a37",
        "R1": "#3987e5",
        "R2": "#d95926",
        "R3": "#199e70",
        "R1_final": "#9085e9",
    },
}

LABEL = {
    "R1": "R1 (attention input basis)",
    "R2": "R2 (MLP input basis)",
    "R3": "self_attn.R2 (head-dim, R3)",
    "R1_final": "R1_final",
}


# ----------------------------------------------------------------------------- 지표 계산
_HADAMARD_CACHE = {}


def get_init_hadamard(size: int) -> torch.Tensor:
    """초기화에 쓰인 결정론적 Walsh-Hadamard 행렬 (optimize_rotation.py 와 동일 함수)."""
    if size not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[size] = hadamard_matrix(size, "cpu").to(torch.float64)
    return _HADAMARD_CACHE[size]


def compare_to_init(R: torch.Tensor, H: torch.Tensor, sign_align: bool = False) -> dict:
    """R 과 초기값 H 의 차이를 요약. sign_align=True 면 random_hadamard 의 행 부호를 복원한다."""
    R = R.to(torch.float64)
    n = R.shape[0]

    row_dot = (R * H).sum(dim=1)
    if sign_align:
        s = torch.sign(row_dot)
        s[s == 0] = 1.0
        H = H * s.unsqueeze(1)  # diag(s) @ H  (random_hadamard_matrix 의 형태)
        row_dot = (R * H).sum(dim=1)

    row_cos = row_dot / (R.norm(dim=1) * H.norm(dim=1)).clamp_min(1e-12)
    frob = (R - H).norm()

    return {
        "cos_matrix": float((R * H).sum() / (R.norm() * H.norm())),
        "row_cos_mean": float(row_cos.mean()),
        "row_cos_min": float(row_cos.min()),
        "row_cos_max": float(row_cos.max()),
        "angle_deg_mean": float(torch.rad2deg(torch.arccos(row_cos.clamp(-1, 1))).mean()),
        "frob": float(frob),
        "rel_frob": float(frob / H.norm()),
        "size": n,
    }


def orthogonality_error(R: torch.Tensor, device: torch.device) -> float:
    """||R^T R - I||_F / sqrt(n). 학습이 Stiefel manifold 를 벗어났는지 확인용."""
    A = R.to(device=device, dtype=torch.float32)
    n = A.shape[0]
    err = (A.T @ A - torch.eye(n, device=device)).norm()
    return float(err / (n ** 0.5))


def transition_error(A: torch.Tensor, B: torch.Tensor, device: torch.device) -> float:
    """residual 전이 행렬 T = A^T B 가 I 에서 얼마나 벗어났는지 (||T - I||_F / sqrt(n)).

    ReSpinQuant 는 초기에 A = B = H 라서 T = I 이고, 학습이 진행되며 Delta_T = T - I 가
    생긴다. 이 값이 커질수록 residual 을 저랭크로 보정한다는 가정이 약해진다.
    """
    A = A.to(device=device, dtype=torch.float32)
    B = B.to(device=device, dtype=torch.float32)
    n = A.shape[0]
    T = A.T @ B
    return float((T - torch.eye(n, device=device)).norm() / (n ** 0.5))


# ----------------------------------------------------------------------------- R.bin 파싱
def parse_rotations(state: dict) -> dict:
    """R.bin 의 key 를 종류별로 정리한다."""
    per_layer = {}   # layer_idx -> {"R1": tensor, "R2": tensor, "R3": tensor}
    globals_ = {}    # "R1_final" / "R1" (SpinQuant 전역 R1)

    re_head = re.compile(r"layers\.(\d+)\.self_attn\.R2$")
    re_res = re.compile(r"layers\.(\d+)\.(R1|R2)$")

    for key, value in state.items():
        key = key.replace(".weight", "")
        value = value.detach().to("cpu").to(torch.float64)

        m = re_head.search(key)
        if m:
            per_layer.setdefault(int(m.group(1)), {})["R3"] = value
            continue
        m = re_res.search(key)
        if m:
            per_layer.setdefault(int(m.group(1)), {})[m.group(2)] = value
            continue
        if key.endswith("R1_final"):
            globals_["R1_final"] = value
        elif key.endswith("R1"):
            globals_["R1"] = value  # SpinQuant 전역 R1
        else:
            print(f"[warn] 인식하지 못한 key 를 건너뜁니다: {key}")

    return {"per_layer": per_layer, "globals": globals_}


def build_records(rot: dict, device: torch.device, do_orth: bool) -> list:
    """(kind, layer, chain_idx, metrics...) 레코드 리스트를 만든다."""
    per_layer, globals_ = rot["per_layer"], rot["globals"]
    num_layers = (max(per_layer) + 1) if per_layer else 0
    records = []

    def add(kind, layer, chain_idx, R, sign_align):
        H = get_init_hadamard(R.shape[0])
        rec = {"kind": kind, "layer": layer, "chain_idx": chain_idx,
               "sign_aligned": int(sign_align)}
        rec.update(compare_to_init(R, H, sign_align=sign_align))
        rec["orth_err"] = orthogonality_error(R, device) if do_orth else float("nan")
        records.append(rec)

    for i in range(num_layers):
        entry = per_layer.get(i, {})
        # residual stream 을 따라 R1 -> R2 순으로 쌓이므로 chain index 는 2i, 2i+1
        if "R1" in entry:
            add("R1", i, 2 * i, entry["R1"], sign_align=False)
        if "R2" in entry:
            add("R2", i, 2 * i + 1, entry["R2"], sign_align=False)
        if "R3" in entry:
            # random_hadamard_matrix 로 초기화 -> 행 부호를 복원해서 비교
            add("R3", i, i, entry["R3"], sign_align=True)

    if "R1_final" in globals_:
        add("R1_final", num_layers, 2 * num_layers, globals_["R1_final"], sign_align=False)
    if "R1" in globals_:  # SpinQuant 전역 R1 (random_hadamard init)
        add("R1_global", -1, -1, globals_["R1"], sign_align=True)

    return records


def build_transitions(rot: dict, device: torch.device) -> list:
    """residual chain 을 따라 인접한 두 rotation 의 전이 오차 T = A^T B."""
    per_layer = rot["per_layer"]
    globals_ = rot["globals"]
    num_layers = (max(per_layer) + 1) if per_layer else 0

    chain = []  # (name, chain_idx, tensor)
    for i in range(num_layers):
        entry = per_layer.get(i, {})
        if "R1" in entry:
            chain.append((f"L{i}.R1", 2 * i, entry["R1"]))
        if "R2" in entry:
            chain.append((f"L{i}.R2", 2 * i + 1, entry["R2"]))
    if "R1_final" in globals_:
        chain.append(("R1_final", 2 * num_layers, globals_["R1_final"]))

    out = []
    for (na, ia, A), (nb, ib, B) in zip(chain[:-1], chain[1:]):
        out.append({
            "pair": f"{na}->{nb}",
            "chain_idx": ia,
            "transition_err": transition_error(A, B, device),
        })
    return out


# ----------------------------------------------------------------------------- 시각화
def _style_axes(ax, colors):
    ax.set_facecolor(colors["surface"])
    ax.grid(True, color=colors["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["grid"])
    ax.tick_params(colors=colors["muted"], labelsize=9)
    ax.xaxis.label.set_color(colors["muted"])
    ax.yaxis.label.set_color(colors["muted"])


def _series(records, kind, xkey="chain_idx"):
    rows = sorted([r for r in records if r["kind"] == kind], key=lambda r: r[xkey])
    return rows


_EPS = 1e-9  # log scale 에서 0 이 되는 것을 막는 하한


def plot_residual_chain(records, out_path, theme, title_suffix):
    """x축 = 모델을 수직으로 쌓았을 때의 rotation index (L0.R1, L0.R2, L1.R1, ..., R1_final).

    학습 후 편차가 1e-2 수준이라 cosine 을 그대로 그리면 전부 1.0 에 붙어버린다.
    그래서 위쪽은 (1 - cosine) 을 log scale 로, 아래쪽은 relative Frobenius 를 log scale 로 그린다.
    (직교행렬이면 rel_frob = sqrt(2(1-cos)) 이라 두 값은 서로의 단조 변환이다.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = PALETTE[theme]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             facecolor=colors["surface"])
    fig.subplots_adjust(hspace=0.18)

    kinds = [k for k in ("R1", "R2") if _series(records, k)]
    finals = _series(records, "R1_final")

    def dev(r):   # 1 - cosine
        return max(1.0 - r["cos_matrix"], _EPS)

    for ax, getter, ylabel in (
        (axes[0], dev, r"$1-\cos(R,\,H)$   (log)"),
        (axes[1], lambda r: max(r["rel_frob"], _EPS), r"$\|R-H\|_F\,/\,\|H\|_F$   (log)"),
    ):
        _style_axes(ax, colors)
        ax.set_yscale("log")
        for kind in kinds:
            rows = _series(records, kind)
            x = [r["chain_idx"] for r in rows]
            ax.plot(x, [getter(r) for r in rows], color=colors[kind], linewidth=2.0,
                    marker="o", markersize=4.5, label=LABEL[kind])
            if getter is dev:  # 가장 많이 돌아간 행(worst axis)을 점선으로 같이 보여준다
                ax.plot(x, [max(1.0 - r["row_cos_min"], _EPS) for r in rows],
                        color=colors[kind], linewidth=1.2, linestyle="--", alpha=0.75,
                        label=f"{LABEL[kind]} — worst row")
        for r in finals:
            ax.scatter([r["chain_idx"]], [getter(r)], color=colors["R1_final"],
                       marker="*", s=170, zorder=5, label=LABEL["R1_final"])
            ax.annotate("R1_final", (r["chain_idx"], getter(r)),
                        textcoords="offset points", xytext=(-6, 10),
                        ha="right", color=colors["muted"], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)

    axes[1].set_xlabel("rotation index along the residual stream "
                       "(L0.R1, L0.R2, L1.R1, ... , R1_final)", fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(colors["text"])

    axes[0].set_title(f"Trained rotation drift from Walsh-Hadamard init{title_suffix}\n"
                      "higher = further from the Hadamard initialization",
                      color=colors["text"], fontsize=12, loc="left", pad=12)
    fig.savefig(out_path, dpi=160, facecolor=colors["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_head_rotation(records, out_path, theme, title_suffix):
    """head_dim rotation(R3) 은 random_hadamard init 이라 행 부호를 맞춘 뒤 비교한 값."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _series(records, "R3", xkey="layer")
    if not rows:
        return
    colors = PALETTE[theme]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                             facecolor=colors["surface"])
    fig.subplots_adjust(hspace=0.18)

    x = [r["layer"] for r in rows]
    for ax, getter, ylabel in (
        (axes[0], lambda r: max(1.0 - r["cos_matrix"], _EPS),
         r"$1-\cos(R,\,H)$  (sign-aligned, log)"),
        (axes[1], lambda r: max(r["rel_frob"], _EPS),
         r"$\|R-H\|_F\,/\,\|H\|_F$  (sign-aligned, log)"),
    ):
        _style_axes(ax, colors)
        ax.set_yscale("log")
        ax.plot(x, [getter(r) for r in rows], color=colors["R3"], linewidth=2.0,
                marker="o", markersize=4.5, label=LABEL["R3"])
        ax.set_ylabel(ylabel, fontsize=10)
    axes[0].plot(x, [max(1.0 - r["row_cos_min"], _EPS) for r in rows],
                 color=colors["R3"], linewidth=1.2, linestyle="--", alpha=0.75,
                 label="worst row")
    axes[1].set_xlabel("layer index", fontsize=10)
    leg = axes[0].legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(colors["text"])
    axes[0].set_title(f"Head-dim rotation (R3) drift from its random-Hadamard init{title_suffix}",
                      color=colors["text"], fontsize=12, loc="left", pad=12)
    fig.savefig(out_path, dpi=160, facecolor=colors["surface"], bbox_inches="tight")
    plt.close(fig)


def plot_transitions(transitions, out_path, theme, title_suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not transitions:
        return
    colors = PALETTE[theme]
    fig, ax = plt.subplots(figsize=(11, 4), facecolor=colors["surface"])
    _style_axes(ax, colors)
    x = [t["chain_idx"] for t in transitions]
    y = [t["transition_err"] for t in transitions]
    ax.plot(x, y, color=colors["R1_final"], linewidth=2.0, marker="o", markersize=4.5,
            label=r"$\|A^{\top}B - I\|_F/\sqrt{n}$ (adjacent rotations)")
    ax.set_xlabel("rotation index along the residual stream", fontsize=10)
    ax.set_ylabel("residual transition error", fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9, loc="best")
    for text in leg.get_texts():
        text.set_color(colors["text"])
    ax.set_title(f"Residual transition T = A^T B deviation from I{title_suffix}",
                 color=colors["text"], fontsize=12, loc="left", pad=12)
    fig.savefig(out_path, dpi=160, facecolor=colors["surface"], bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rotation_path", required=True, help="학습된 R.bin 경로")
    parser.add_argument("--output_dir", default="figures/rotation_drift")
    parser.add_argument("--tag", default="", help="파일 이름/제목에 붙일 태그 (예: llama2-7b)")
    parser.add_argument("--theme", default="both", choices=["light", "dark", "both"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="orthogonality / transition 계산에 쓸 device")
    parser.add_argument("--no_orth", action="store_true", help="||R^T R - I|| 계산 생략")
    parser.add_argument("--no_transition", action="store_true", help="인접 전이 오차 계산 생략")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    title_suffix = f" — {args.tag}" if args.tag else ""

    state = torch.load(args.rotation_path, map_location="cpu", weights_only=False)
    rot = parse_rotations(state)
    n_layers = (max(rot["per_layer"]) + 1) if rot["per_layer"] else 0
    print(f"[info] {args.rotation_path}: {len(state)} tensors, {n_layers} layers, "
          f"globals={list(rot['globals'])}, device={device}")

    records = build_records(rot, device, do_orth=not args.no_orth)
    transitions = [] if args.no_transition else build_transitions(rot, device)

    # ---- CSV
    csv_path = os.path.join(args.output_dir, f"rotation_drift{suffix}.csv")
    fields = ["kind", "layer", "chain_idx", "size", "sign_aligned", "cos_matrix",
              "row_cos_mean", "row_cos_min", "row_cos_max", "angle_deg_mean",
              "frob", "rel_frob", "orth_err"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in fields})
    print(f"[info] wrote {csv_path}")

    if transitions:
        t_path = os.path.join(args.output_dir, f"residual_transition{suffix}.csv")
        with open(t_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["pair", "chain_idx", "transition_err"])
            writer.writeheader()
            writer.writerows(transitions)
        print(f"[info] wrote {t_path}")

    # ---- 요약 출력
    print(f"\n{'kind':<10}{'layer':>6}{'chain':>7}{'cos':>10}{'1-cos':>11}"
          f"{'rel_frob':>10}{'angle°':>9}{'orth_err':>11}")
    for r in records:
        print(f"{r['kind']:<10}{r['layer']:>6}{r['chain_idx']:>7}{r['cos_matrix']:>10.6f}"
              f"{1.0 - r['cos_matrix']:>11.2e}{r['rel_frob']:>10.4f}"
              f"{r['angle_deg_mean']:>9.3f}{r['orth_err']:>11.2e}")

    # ---- 그림
    themes = ["light", "dark"] if args.theme == "both" else [args.theme]
    for theme in themes:
        tsuf = f"{suffix}_{theme}"
        plot_residual_chain(records, os.path.join(args.output_dir, f"residual_chain{tsuf}.png"),
                            theme, title_suffix)
        plot_head_rotation(records, os.path.join(args.output_dir, f"head_rotation{tsuf}.png"),
                           theme, title_suffix)
        plot_transitions(transitions, os.path.join(args.output_dir, f"residual_transition{tsuf}.png"),
                         theme, title_suffix)
    print(f"[info] figures saved under {args.output_dir}")


if __name__ == "__main__":
    main()
