#!/bin/bash

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" )
MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B"  )
# MODELS=( "Qwen/Qwen2.5-7B" )
# GPUS=(5 6 7 )
GPUS=(6 7 )
OUTPUT_BASE="/home/jhkcool97/RotationBasedRepository/outputs"
BIT_CONFIGS=("4,8" "4,4")

for CONFIG in "${BIT_CONFIGS[@]}"; do
    # 쉼표를 기준으로 W_BIT와 A_BIT 분리
    IFS=',' read -r W_BIT A_BIT <<< "$CONFIG"
    
    echo "========================================"
    echo "  Starting Experiments for W${W_BIT}A${A_BIT}"
    echo "========================================"

    # 1. All Rotations 적용
    for i in "${!MODELS[@]}"; do
        GPU_ID=${GPUS[$i]}
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
        mkdir -p "$TARGET_DIR"

        CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16 > "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_ALL_Rotations.txt" 2>&1 &
    done
    wait

    # 2. No R4 (deactivate_r1)
    for i in "${!MODELS[@]}"; do
        GPU_ID=${GPUS[$i]}
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --deactivate_r1 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16 > "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R4.txt" 2>&1 &
    done
    wait

    # 3. No R2 (deactivate_r2)
    for i in "${!MODELS[@]}"; do
        GPU_ID=${GPUS[$i]}
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --deactivate_r2 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16 > "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R2.txt" 2>&1 &
    done
    wait

    # 4. No R1 (deactivate_r1 + online_r2)
    for i in "${!MODELS[@]}"; do
        GPU_ID=${GPUS[$i]}
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --deactivate_r1 --online_r2 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16 > "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R1.txt" 2>&1 &
    done
    wait
done

# for i in "${!MODELS[@]}"; do
#     GPU_ID=${GPUS[$i]}
#     MODEL_PATH=${MODELS[$i]}
#     MODEL_NAME=$(basename "$MODEL_PATH")

#     TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
#     mkdir -p "$TARGET_DIR"
#     echo "Running $MODEL_PATH on GPU $GPU_ID"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
#         --input_model "$MODEL_PATH" \
#         --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
#         --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
#         --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate  --online_r2  --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16 > "${TARGET_DIR}/log__W${W_BIT}A${A_BIT}_ALL_Rotations.txt" 2>&1 &
    
#     PIDS+=($!)
# done

# wait

# #2. R4를 적용하지 않는 경우  경우
# for i in "${!MODELS[@]}"; do
#     GPU_ID=${GPUS[$i]}
#     MODEL_PATH=${MODELS[$i]}
#     MODEL_NAME=$(basename "$MODEL_PATH")

#     TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
#     mkdir -p "$TARGET_DIR"

#     echo "Running $MODEL_PATH on GPU $GPU_ID"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
#         --input_model "$MODEL_PATH" \
#         --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
#         --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
#         --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate  --online_r2 --deactivate_r1 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16> "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R4.txt" 2>&1 &
    
#     PIDS+=($!)
# done

# wait

# #3. R2를 적용하지 않는 경우
# for i in "${!MODELS[@]}"; do
#     GPU_ID=${GPUS[$i]}
#     MODEL_PATH=${MODELS[$i]}
#     MODEL_NAME=$(basename "$MODEL_PATH")

#     TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
#     mkdir -p "$TARGET_DIR"

#     echo "Running $MODEL_PATH on GPU $GPU_ID"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
#         --input_model "$MODEL_PATH" \
#         --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
#         --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
#         --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate  --deactivate_r2 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16> "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R2.txt" 2>&1 &
    
#     PIDS+=($!)
# done

# wait

# #4. R1를 적용하지 않는 경우
# for i in "${!MODELS[@]}"; do
#     GPU_ID=${GPUS[$i]}
#     MODEL_PATH=${MODELS[$i]}
#     MODEL_NAME=$(basename "$MODEL_PATH")

#     TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
#     mkdir -p "$TARGET_DIR"

#     echo "Running $MODEL_PATH on GPU $GPU_ID"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py \
#         --input_model "$MODEL_PATH" \
#         --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
#         --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
#         --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
#         --rotate  --deactivate_r1 --online_r2 --wikitext2 --w_rtn --w_clip --lm_eval --lm_eval_batch_size 16> "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R2.txt" 2>&1 &
    
#     PIDS+=($!)
# done

# wait
