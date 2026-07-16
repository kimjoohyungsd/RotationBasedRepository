#!/bin/bash
cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$ 
    exit 1
}
trap cleanup SIGINT

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" )
MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-3.1-8B"  )
# MODELS=( "Qwen/Qwen2.5-7B" )
GPUS=(5 6 7 )
# GPUS=(0 1)
VISIBLE_GPUS=$(IFS=,; echo "${GPUS[*]}")
echo "Visible GPUs: ${VISIBLE_GPUS}"
OUTPUT_BASE="/home/jhkcool97/RotationBasedRepository/logs/Permutations"
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

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_ALL_Rotations.txt" 
    done
    wait

    # 2. zigzag permutation + Block Hadamard Rotation
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
        mkdir -p "$TARGET_DIR"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --permute --permute_mode 'zigzag' --diagonal_size 128 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_zigzag_permute.txt" 
    done
    wait

    # 3. MassDiff Permutation + Block hadamard Rotation
    for i in "${!MODELS[@]}"; do
        # GPU_ID=${GPUS[$i]}
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
        mkdir -p "$TARGET_DIR"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --permute --distribute --permute_mode 'massdiff' --diagonal_size 128 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_massdiff_permute.txt" 
    done
    wait


    # 4. Random Permutation + Block hadamard Rotation
    for i in "${!MODELS[@]}"; do
        MODEL_PATH=${MODELS[$i]}
        MODEL_NAME=$(basename "$MODEL_PATH")
        TARGET_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
        mkdir -p "$TARGET_DIR"

        CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" python ptq.py \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 4 --model_max_length 2048 --distribute --fp16 False --bf16 True --save_safetensors False \
            --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
            --a_asym --k_asym --v_asym --k_groupsize 128 --v_groupsize 128 \
            --rotate --online_r2 --permute --distribute --permute_mode 'random' --diagonal_size 128 --wikitext2 --w_rtn --w_clip  --eval_out_path "${TARGET_DIR}/log_W${W_BIT}A${A_BIT}_random_permute.txt" 
    done
    wait
done
wait
