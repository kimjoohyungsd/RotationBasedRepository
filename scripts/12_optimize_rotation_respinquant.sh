#!/bin/bash
# coding=utf-8
# ReSpinQuant rotation training.
#
# Learns 2L+1 DENSE D x D residual-stream bases (layers.i.R1, layers.i.R2,
# R1_final) plus the per-layer head rotation R3, all on the Stiefel manifold via
# Cayley-SGD.  The residual transitions T = R_in^T R_out are compressed AFTER
# training, by the truncated SVD in
# eval_utils/rotation_utils.compute_residual_subspace (see --residual_rank in
# scripts/22_eval_ptq_respinquant.sh).
#
# Parameter count: (2L+1) * D^2  -- ~1.09B for LLaMA-3 8B, hence FSDP.
#
# Usage:  bash scripts/12_optimize_rotation_respinquant.sh <model> <w_bits> <a_bits> <kv_bits>
#   e.g.  bash scripts/12_optimize_rotation_respinquant.sh meta-llama/Llama-3.1-8B 16 4 4
#
# Evaluate with scripts/22_eval_ptq_respinquant.sh.
set -euo pipefail

MODEL=${1:?usage: $0 <model> <w_bits> <a_bits> <kv_bits>}
W_BITS=${2:-16}
A_BITS=${3:-4}
KV_BITS=${4:-4}

MODEL_TAG=$(basename "$MODEL")
OUT_ROT=${OUT_ROT:-/home/jhkcool97/Rotation_repository/Matrixes/${MODEL_TAG}/ReSpinQuant/W:${W_BITS}A:${A_BITS}KV:${KV_BITS}}
RUN_TAG=${RUN_TAG:-Training/${MODEL_TAG}/ReSpinQuant_w${W_BITS}a${A_BITS}kv${KV_BITS}}
NPROC=${NPROC:-8}

echo "=== ReSpinQuant rotation training ==="
echo "  model     : ${MODEL}"
echo "  bits      : w${W_BITS} a${A_BITS} kv${KV_BITS}"
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
--learning_rate 15 \
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
--respinquant \
--k_groupsize 128 \
--v_groupsize 128

# --learning_rate 15 is SpinQuant's Stiefel/Cayley-SGD step size. In this mode
# EVERY trainable parameter lives on the manifold, so one lr is correct. Do not
# carry it over to LieReSpinQuant -- see 13_optimize_rotation_lierespinquant.sh.
#
# For the 8B model add FSDP (as in 11_optimize_rotation_fsdp.sh):
#   --fsdp "full_shard auto_wrap offload" \
#   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
#   --fsdp_config scripts/fsdp_config.json
