#!/bin/bash

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" )
MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"  )
# MODELS=( "Qwen/Qwen2.5-7B" )
# GPUS=(5 6 7 )
GPUS=(5 6 )
# GPUS=( 2) # 사용할 GPU 번호들
#1. R4를 적용하지 않는 모듈을 활용하는 경우
W_BIT=4  # Weight bit-width
A_BIT=8  # Activation bit-width
OUTPUT_BASE="/home/jhkcool97/RotationBasedRepository/outputs"

for i in "${!MODELS[@]}"; do
    GPU_ID=${GPUS[$i]}
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "Running $MODEL_PATH on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
        --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate  --online_r2 --deactivate_r1 --wikitext2 --w_rtn --w_clip > "${OUTPUT_BASE}/${MODEL_NAME}/log__W${W_BIT}A${A_BIT}.txt" 2>&1 &
    
    PIDS+=($!)
done

wait

#2. R4를 적용하는 모듈을 활용하는 경우
for i in "${!MODELS[@]}"; do
    GPU_ID=${GPUS[$i]}
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=$(basename "$MODEL_PATH")
    echo "Running $MODEL_PATH on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits 4 --a_bits 4 --k_bits 16 --v_bits 16 \
        --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate  --online_r2 --wikitext2 --w_rtn --w_clip > "/home/jhkcool97/RotationBasedRepository/outputs/log_${MODEL_NAME}_{R1}.txt" 2>&1 &
    
    PIDS+=($!)
done

wait