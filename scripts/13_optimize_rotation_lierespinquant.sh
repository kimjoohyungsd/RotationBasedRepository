#!/bin/bash
# coding=utf-8
# LieReSpinQuant rotation training.
#
# Instead of learning 2L+1 dense bases and compressing the residual transitions
# with SVD afterwards, this learns each transition DIRECTLY as a low-rank
# skew-symmetric Cayley rotation:
#
#     bases[0]   = Hadamard (fixed)
#     bases[k+1] = bases[k] @ dR_k
#     dR_k       = Cayley(U_k diag(g_k) V_k^T - V_k diag(g_k) U_k^T) = I + P_k Z_k P_k^T
#
# so bases[k]^T bases[k+1] IS dR_k -- exactly orthogonal, always in SO(D), and
# already rank-2r at inference. Nothing is approximated post-hoc.
#
# Parameter count: 2L * (2 D r + r) -- ~16.8M at D=4096, L=32, r=32, i.e. ~65x
# fewer than ReSpinQuant's ~1.09B, so FSDP is usually unnecessary.
#
# Usage:  bash scripts/13_optimize_rotation_lierespinquant.sh <model> <w_bits> <a_bits> <kv_bits>
#   e.g.  bash scripts/13_optimize_rotation_lierespinquant.sh meta-llama/Llama-3.1-8B 16 4 4
#
# Knobs (override via environment):
#   LIE_RANK       r, learned rotation planes per transition (generator rank <= 2r)
#   LIE_LR         step size for U/V/gamma  -- Euclidean parameters, NOT Stiefel
#   LIE_GATE_L1    L1 on the gates; > 0 lets each transition pick its own
#                  effective rank instead of a uniform r
#   LIE_GATE_INIT  std of the initial gates; near 0 starts from plain SpinQuant
#
# Evaluate with scripts/23_eval_ptq_lierespinquant.sh (match --lie_rank).
set -euo pipefail

MODEL=${1:?usage: $0 <model> <w_bits> <a_bits> <kv_bits>}
W_BITS=${2:-16}
A_BITS=${3:-4}
KV_BITS=${4:-4}

LIE_RANK=${LIE_RANK:-32}
LIE_LR=${LIE_LR:-1e-3}
LIE_GATE_L1=${LIE_GATE_L1:-0.0}
LIE_GATE_INIT=${LIE_GATE_INIT:-1e-2}
R3_LR=${R3_LR:-15}

MODEL_TAG=$(basename "$MODEL")
OUT_ROT=${OUT_ROT:-/home/jhkcool97/Rotation_repository/Matrixes/${MODEL_TAG}/LieReSpinQuant/W:${W_BITS}A:${A_BITS}KV:${KV_BITS}}
RUN_TAG=${RUN_TAG:-Training/${MODEL_TAG}/LieReSpinQuant_r${LIE_RANK}_w${W_BITS}a${A_BITS}kv${KV_BITS}}
NPROC=${NPROC:-8}

echo "=== LieReSpinQuant rotation training ==="
echo "  model     : ${MODEL}"
echo "  bits      : w${W_BITS} a${A_BITS} kv${KV_BITS}"
echo "  lie rank  : ${LIE_RANK}  (generator rank <= $((2 * LIE_RANK)))"
echo "  lie lr    : ${LIE_LR}   (Euclidean)      R3 lr: ${R3_LR} (Stiefel)"
echo "  gate L1   : ${LIE_GATE_L1}   gate init: ${LIE_GATE_INIT}"
echo "  rotations : ${OUT_ROT}/R.bin"
echo "  gpus      : ${NPROC}"

torchrun --nnodes=1 --nproc_per_node="${NPROC}" optimize_rotation.py \
--input_model "$MODEL" \
--output_rotation_path "$OUT_ROT" \
--output_dir "${RUN_TAG}/" \
--logging_dir "${RUN_TAG}_log/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate "$R3_LR" \
--lie_learning_rate "$LIE_LR" \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--max_steps 100 \
--w_bits "$W_BITS" \
--a_bits "$A_BITS" \
--k_bits "$KV_BITS" \
--v_bits "$KV_BITS" \
--w_rtn \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--lierespinquant \
--lie_rank "$LIE_RANK" \
--lie_gate_l1 "$LIE_GATE_L1" \
--lie_gate_init "$LIE_GATE_INIT" \
--k_groupsize 128 \
--v_groupsize 128

# ── Why two learning rates ──────────────────────────────────────────────────
# --learning_rate (15) is SpinQuant's Stiefel/Cayley-SGD step size and applies
# only to R3, which still lives on the manifold. U/V/gamma are ordinary
# Euclidean parameters initialised at scale 1/sqrt(D) ~ 0.016 -- feeding them
# lr=15 diverges on the first step. optimize_rotation.py puts them in a separate
# non-Stiefel param group driven by --lie_learning_rate.
#
# ── Suggested sweeps ────────────────────────────────────────────────────────
#   rank:            LIE_RANK=8 / 32 / 128   (ReSpinQuant Table 5 shows strong
#                                             rank sensitivity; here rank is the
#                                             training budget, not a post-hoc cut)
#   adaptive rank:   LIE_GATE_L1=1e-4        then read the "effective ranks per
#                                             transition" line in the train log
#                                             to see where the model spends rank
#   identity start:  LIE_GATE_INIT=0 gives dR = I exactly, i.e. the run begins
#                    from plain SpinQuant (one global rotation) -- but U/V then
#                    receive zero gradient, so keep it small-nonzero (default).
#
# For the 8B model FSDP is optional here (only ~16.8M trainable params), but if
# activation memory is the constraint, add:
#   --fsdp "full_shard auto_wrap offload" \
#   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#   --fsdp_config scripts/fsdp_config.json
