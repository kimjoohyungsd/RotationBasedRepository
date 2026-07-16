#!/bin/bash
cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$ 
    exit 1
}
trap cleanup SIGINT

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" )
MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" )
# MODELS=( "Qwen/Qwen2.5-7B" )
# GPUS=(5 6 7 )
GPUS=(0 1 2)
VISIBLE_GPUS=$(IFS=,; echo "${GPUS[*]}")
echo "Visible GPUs: ${VISIBLE_GPUS}"
OUTPUT_BASE="/home/jhkcool97/RotationBasedRepository/logs/Rotations/"
# BIT_CONFIGS=("4,8" "4,4")

# BIT_CONFIGS=("4,8" )
BIT_CONFIGS=("4,4")
for CONFIG in "${BIT_CONFIGS[@]}"; do
    # 쉼표를 기준으로 W_BIT와 A_BIT 분리
    IFS=',' read -r W_BIT A_BIT <<< "$CONFIG"
    
    echo "========================================"
    echo "  Starting Experiments for W${W_BIT}A${A_BIT}"
    echo "========================================"

    echo "Running Experiment 1: All Rotations (W${W_BIT}A${A_BIT})"
    # 1. All Rotations 적용
    for i in "${!MODELS[@]}"; do
        
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
        mkdir -p "$TARGET_DIR"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_ALL_Rotations.txt" 
    done
    wait

    

    echo "Running Experiment 2: No R4 (W${W_BIT}A${A_BIT})"
    # 2. No R4 (deactivate_r1)
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R4.txt" 
    done
    wait

    echo "Running Experiment 3: No R2 (W${W_BIT}A${A_BIT})"
    # 3. No R2 (deactivate_r2)
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --deactivate_r2 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R2.txt"
    done
    wait

    

    echo "Running Experiment 4: No R1 (W${W_BIT}A${A_BIT})"
    # 4. No R1 (deactivate_r1 + online_r2)
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --deactivate_r1 --online_r2 --wikitext2 --w_rtn --w_clip --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R1.txt" 2>&1 &
    done
    wait

    echo "Running Experiment 5: Naive Round to Nearest Quantization (W${W_BIT}A${A_BIT})"
    # 5. No R1 (deactivate_r1 + online_r2)
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
             --wikitext2 --w_rtn --w_clip --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_RTN.txt" 2>&1 &
    done
    wait

    # echo "Running Experiment 6: No R4, R2 (W${W_BIT}A${A_BIT})"
    # # 6. No R4, R2 (only R1)
    # for i in "${!MODELS[@]}"; do
    #     MODEL_PATH=${MODELS[$i]}
    #     MODEL_NAME=$(basename "$MODEL_PATH")
    #     TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"

    #     CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq_qwen3.py \
    #         --input_model "$MODEL_PATH" \
    #         --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #         --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #         --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
    #         --rotate --deactivate_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_NO_R4_R2.txt" 
    # done
    # wait
done