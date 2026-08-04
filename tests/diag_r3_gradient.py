# coding=utf-8
"""왜 --lierespinquant 로 학습하면 self_attn.R2 (R3) 가 안 움직이는가?

optimize_rotation.py 와 동일한 순서로 작은 모델을 세우고
  (a) R3 가 requires_grad 인지,
  (b) backward 후 .grad 가 들어오는지,
  (c) SGDG.step() 후 실제로 값이 변하는지
를 lie 모드와 respinquant 모드에서 각각 찍는다.

    python tests/diag_r3_gradient.py
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers  # noqa: E402

from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant  # noqa: E402
from train_utils.optimizer import SGDG  # noqa: E402
from utils.hadamard_utils import hadamard_matrix, random_hadamard_matrix  # noqa: E402
from utils.lie_rotation import LieBasisChain  # noqa: E402


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super().__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32))


def build(mode, device, hidden=128, layers=2, heads=4, rank=8):
    cfg = transformers.LlamaConfig(
        hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=heads, vocab_size=64, max_position_embeddings=64,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = LlamaForCausalLMQuant(cfg).to(device=device, dtype=torch.float32)
    model.eval()

    # optimize_rotation.py 와 같은 순서: 먼저 전부 freeze, 그 다음 rotation 모듈 생성
    for p in model.parameters():
        p.requires_grad = False

    head_dim = hidden // heads
    cfg.lie_gate_l1 = 0.0
    if mode == "lie":
        cfg.respinquant = True          # 현재 optimize_rotation.py 는 여기가 False 다 (아래 주석 참고)
        cfg.lierespinquant = True
        model.model.lie_chain = LieBasisChain(
            dim=hidden, num_layers=layers, rank=rank,
            base=hadamard_matrix(hidden, device), gate_init=1e-2, device=device,
        ).to(device)
        lie_params = model.model.lie_chain.trainable_parameters()
    else:
        cfg.respinquant = True
        cfg.lierespinquant = False
        for i in range(layers):
            model.model.layers[i].R1 = RotateModule(hadamard_matrix(hidden, device)).to(device)
            model.model.layers[i].R2 = RotateModule(hadamard_matrix(hidden, device)).to(device)
        model.model.R1_final = RotateModule(hadamard_matrix(hidden, device)).to(device)
        lie_params = ([model.model.layers[i].R1.weight for i in range(layers)]
                      + [model.model.layers[i].R2.weight for i in range(layers)]
                      + [model.model.R1_final.weight])

    head_params = []
    for i in range(layers):
        m = RotateModule(random_hadamard_matrix(head_dim, device)).to(device)
        model.model.layers[i].self_attn.R2 = m
        head_params.append(m.weight)
    return model, lie_params, head_params


def report(mode, device):
    print(f"\n===== mode = {mode} =====")
    model, lie_params, head_params = build(mode, device)

    print("  requires_grad:  R3 =", [p.requires_grad for p in head_params],
          " other =", [p.requires_grad for p in lie_params][:2], "...")

    ids = torch.randint(0, 64, (2, 16), device=device)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    def g(p):
        if p.grad is None:
            return "None"
        return f"{p.grad.abs().max().item():.3e}"

    print("  |grad|max  R3    :", [g(p) for p in head_params])
    print("  |grad|max  other :", [g(p) for p in lie_params][:3], "...")

    if mode == "lie":
        opt = SGDG([
            {"params": lie_params, "stiefel": False, "lr": 1e-3},
            {"params": head_params, "stiefel": True, "lr": 15},
        ], lr=15, stiefel=True)
    else:
        opt = SGDG(lie_params + head_params, lr=15, stiefel=True)

    before = [p.detach().clone() for p in head_params]
    before_other = [p.detach().clone() for p in lie_params]
    opt.step()
    d_r3 = [(p - b).abs().max().item() for p, b in zip(head_params, before)]
    d_ot = [(p - b).abs().max().item() for p, b in zip(lie_params, before_other)]
    print("  step() delta R3   :", [f"{d:.3e}" for d in d_r3])
    print("  step() delta other:", [f"{d:.3e}" for d in d_ot][:3], "...")
    moved = all(d > 0 for d in d_r3)
    print(f"  --> R3 {'MOVED' if moved else 'DID NOT MOVE'}")
    return moved


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"R3 gradient diagnostic ({device})")
    lie_moved = report("lie", device)
    respin_moved = report("respinquant", device)
    print("\nsummary: lie R3 moved =", lie_moved, "| respinquant R3 moved =", respin_moved)


if __name__ == "__main__":
    main()
