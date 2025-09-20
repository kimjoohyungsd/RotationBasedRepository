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
--output_rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/  \
--output_dir "R1644_g_size_32/" \
--logging_dir "R1644_g_size_32_log/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 1 \
--logging_steps 1 \
--learning_rate 1.5 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--max_steps 100 \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_rtn \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--a_groupsize 32 \
--k_groupsize 128 \
--v_groupsize 128 \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
# --fsdp_config /home/jhkcool97/SpinQuant/scripts/fsdp_config.json \

# --fsdp_offload_params True \

