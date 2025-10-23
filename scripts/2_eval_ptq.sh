# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.
export MASTER_PORT=$((12000 + $RANDOM % 20000))
# Default Option
# torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT ptq.py \
# --input_model $1 \
# --do_train False \
# --do_eval True \
# --per_device_eval_batch_size 4 \
# --model_max_length 2048 \
# --fp16 False \
# --bf16 True \
# --save_safetensors False \
# --w_bits $2 \
# --a_bits $3 \
# --k_bits $4 \
# --v_bits $4 \
# --w_clip \
# --a_asym \
# --k_asym \
# --v_asym \
# --k_groupsize 128 \
# --v_groupsize 128 \
# --wikitext2 \
# --w_rtn
# --rotate \
# --optimized_rotation_path "your_path/R.bin" \

# Option 1 (Default running with Rotation Matrix Fusion)
# torchrun --nnodes=1 --nproc_per_node=1 ptq.py \
# --input_model $1 \
# --do_train False \
# --do_eval True \
# --per_device_eval_batch_size 4 \
# --model_max_length 2048 \
# --fp16 False \
# --bf16 True \
# --save_safetensors False \
# --w_bits $2 \
# --a_bits $3 \
# --k_bits $4 \
# --v_bits $4 \
# --k_asym \
# --v_asym \
# --k_groupsize 128 \
# --v_groupsize 128 \
# --rotate \
# --optimized_rotation_path "/home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R1644.bin" \
# --wikitext2 \
# --w_clip \
# # --w_groupsize 32 \
# # --a_groupsize 32 \
# # --w_rtn \
# # --a_asym \





# Option 2 (Only with Model)
torchrun --nnodes=1 --nproc_per_node=1 --master_port=$MASTER_PORT ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize -1 \
--a_groupsize -1 \
--k_asym \
--v_asym \
--wikitext2 \
--w_clip \
--w_rtn \
--rotate \
--diagonal \
--diagonal_size 512 \
# --offline  \

# --smooth_quant \
# --alpha 0.65 \
# --attention \



# --a_asym \



# --optimized_rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R1644.bin



# --lm_eval_dat "arc_challenge" \
# --eval_out_path /home/jhkcool97/SpinQuant/Results/Zeroshot/Arc_challenge/W4A4KV4/W,A:32,KV:head/RTN/Result.txt \
















# --optimized_rotation_path /home/jhkcool97/Rotati