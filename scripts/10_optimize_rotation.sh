# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
torchrun --nnodes=1 --nproc_per_node=7 optimize_rotation.py \
--input_model $1  \
--output_rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/ \
--output_dir "R1644_g_size_32/" \
--logging_dir "R1644_g_size_32_log/" \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--save_safetensors False \
--max_steps 100 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--k_asym \
--v_asym \
--a_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize 32 \
--a_groupsize 32
