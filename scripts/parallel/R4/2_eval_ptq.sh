#!/bin/bash

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" )
MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"  )
# MODELS=( "Qwen/Qwen2.5-7B" )
# GPUS=(5 6 7 )
GPUS=(6 7 )
# GPUS=( 2) # 사용할 GPU 번호들
#1. R4를 적용하지 않는 모듈을 활용하는 경우
W_BIT=4  # Weight bit-width
A_BIT=8  # Activation bit-width
OUTPUT_BASE="/home/jhkcool97/RotationBasedRepository/outputs"

for i in "${!MODELS[@]}"; do
    GPU_ID=${GPUS[$i]}
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=$(basename "$MODEL_PATH")

    TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    mkdir -p "$TARGET_DIR"
    echo "Running $MODEL_PATH on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
        --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate  --online_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip > "${TARGET_DIR}/log__W${W_BIT}A${A_BIT}_No_R4.txt" 2>&1 &
    
    PIDS+=($!)
done

wait

#2. R4를 적용하는 모듈을 활용하는 경우
for i in "${!MODELS[@]}"; do
    GPU_ID=${GPUS[$i]}
    MODEL_PATH=${MODELS[$i]}
    MODEL_NAME=$(basename "$MODEL_PATH")

    TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
    mkdir -p "$TARGET_DIR"

    echo "Running $MODEL_PATH on GPU $GPU_ID"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
        --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate  --online_r2 --wikitext2 --w_rtn --w_clip > "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_R4.txt" 2>&1 &
    
    PIDS+=($!)
done

wait