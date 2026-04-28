#!/bin/bash
#1. Llama2-70B의 모델로 실행을 하는 경우
MODEL_PATH="meta-llama/Llama-2-70b-hf"
MODEL_NAME=$(basename "$MODEL_PATH")

# 1. Meta-llama에서 R4 Option을 실행하고 실험을 진행
echo "Starting Experiment 1: R4 Option"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
        --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate --distribute  --online_r2 --wikitext2 --w_rtn > "/home/jhkcool97/RotationBasedRepository/outputs/log_${MODEL_NAME}_{R4}.txt" 2>&1

echo "Experiment 1 finished. Starting Experiment 2: No R4 Option"

# 2. Meta-llama에서 R4 Option을 실행하고 실험을 진행
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
        --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate --distribute  --online_r2 --deactivate_r4 --wikitext2 --w_rtn > "/home/jhkcool97/RotationBasedRepository/outputs/log_${MODEL_NAME}_{No_R4}.txt" 2>&1 
