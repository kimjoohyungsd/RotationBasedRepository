# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 4 \
--a_bits 8 \
--w_clip \
--w_groupsize 32 \
--a_asym \
--rotate \
--wikitext2 \
--save_qmodel_path "/home/jhkcool97/SpinQuant/Executorch/consolidated.00.pth" \
--export_to_et

# --optimized_rotation_path "/home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R1644.bin" \
