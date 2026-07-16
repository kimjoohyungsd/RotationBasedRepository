#!/bin/bash

# 1. 인자 확인
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CUDA_VISIBLE_DEVICES> <MODEL_PATH>"
    echo "Example: $0 0,1,2,3 meta-llama/Llama-2-7b-hf"
    exit 1
fi

cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$ 
    exit 1
}

trap cleanup SIGINT

# 2. 인자 할당 및 실행 파일 결정
CUDA_DEVICES=$1
MODEL_PATH=$2
MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="/home/jhkcool97/RotationBasedRepository/logs/${MODEL_NAME}"

# [추가] Qwen3 포함 여부에 따라 실행 스크립트 결정
if [[ "$MODEL_PATH" == *"Qwen3"* ]] || [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
    PY_SCRIPT="ptq_qwen3.py"
    echo ">>> Qwen3 model detected. Using $PY_SCRIPT"
else
    PY_SCRIPT="ptq.py"
    echo ">>> Standard model detected. Using $PY_SCRIPT"
fi

# -------------------------------------------------------------------------
# [상위 루프] 실험할 비트 조합 리스트 (Weight,Activation)
# BIT_CONFIGS=("16,16" "4,8" "4,4")
BIT_CONFIGS=("4,4")
for CONFIG in "${BIT_CONFIGS[@]}"; do
    # 쉼표를 기준으로 W_BIT와 A_BIT 분리
    IFS=',' read -r W_BIT A_BIT <<< "$CONFIG"

    echo "================================================================"
    echo "  Starting All Experiments for W${W_BIT}A${A_BIT} on Model: ${MODEL_NAME}"
    echo "  Executing: $PY_SCRIPT"
    echo "================================================================"

    # 디렉토리 생성 확인
    mkdir -p "$OUTPUT_DIR"

    # -------------------------------------------------------------------------
    # [분기 처리] W16A16 (Full Precision) 베이스라인 측정인 경우
    # -------------------------------------------------------------------------

    if [ "$W_BIT" -eq 16 ] && [ "$A_BIT" -eq 16 ]; then
        echo "Running Baseline Experiment: Full Precision FP16/BF16 (W16A16)"
        
        # 양자화 및 회전 관련 플래그를 모두 제거하고 순수 모델 아키텍처 상태로 평가
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
            --input_model "$MODEL_PATH" \
            --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
            --w_bits 16 --a_bits 16 --k_bits 16 --v_bits 16 --distribute \
            --wikitext2 --eval_out_path "${OUTPUT_DIR}/log_W16A16_Full_Precision.txt"
            
        echo "Baseline experiment for W16A16 finished. Moving to next config."
        continue # 16비트는 소거법 실험이 필요 없으므로 다음 BIT_CONFIGS로 넘어감
    fi

    # # # 실험 1: 모든 Option 실행
    # echo "Running Experiment 1: All Rotations (W${W_BIT}A${A_BIT})"
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 8 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --rotate --distribute --online_r2 --wikitext2  --w_rtn --w_clip \
    #     --eval_out_path "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_All_Rotations.txt" 

    # # 실험 2: No R4 Option 실행
    # echo "Running Experiment 2: No R4 (W${W_BIT}A${A_BIT})"
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --rotate --distribute --online_r2 --deactivate_r4 --wikitext2  --w_rtn --w_clip \
    #     --eval_out_path "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R4.txt" 2>&1

    # # # 실험 3: No R2 Option 실행
    # echo "Running Experiment 3: No R2 (W${W_BIT}A${A_BIT})"
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --rotate --distribute --deactivate_r2 --wikitext2 --w_rtn --w_clip \
    #     --eval_out_path "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R2.txt"

    # # # 실험 3-1: No R2 Option 실행
    # # echo "Running Experiment 3-1: No R2_R4 (W${W_BIT}A${A_BIT})"
    # # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    # #     --input_model "$MODEL_PATH" \
    # #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    # #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    # #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    # #     --rotate --distribute --deactivate_r2 --deactivate_r4 --wikitext2 --w_rtn --w_clip \
    # #     > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R2_R4.txt" 2>&1

    # # 실험 4: No R1 Option 실행
    # echo "Running Experiment 4: No R1 (W${W_BIT}A${A_BIT})"
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --rotate --distribute --deactivate_r1 --online_r2 --wikitext2 --w_rtn --w_clip \
    #     --eval_out_path "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R1.txt" 2>&1

    # 실험 5: No R4, R2 Option 실행
    echo "Running Experiment 5: No R4, R2 (W${W_BIT}A${A_BIT})"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
        --input_model "$MODEL_PATH" \
        --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
        --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
        --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
        --rotate --deactivate_r4 --deactivate_r2 \
        --distribute --wikitext2 --w_rtn --w_clip \
        --eval_out_path "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R4,R2.txt"
    # # 실험 4-1: No R4, R1 Option 실행
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python ptq.py \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --rotate --distribute --deactivate_r1 --deactivate_r4 --online_r2 --wikitext2 --w_rtn --w_clip \
    #     > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_R4_R1.txt" 2>&1

    # 실험 5: No R2,R4 Option 실행

    # 실험 6: No Rotations Option 실행
    # echo "Running Experiment 6: No Rotations (W${W_BIT}A${A_BIT})"
    # CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python $PY_SCRIPT \
    #     --input_model "$MODEL_PATH" \
    #     --do_train False --do_eval True --per_device_eval_batch_size 2 --model_max_length 2048 --fp16 False --bf16 True --save_safetensors False \
    #     --w_bits $W_BIT --a_bits $A_BIT --k_bits 16 --v_bits 16 \
    #     --k_asym --v_asym --a_asym --k_groupsize 128 --v_groupsize 128 \
    #     --distribute --wikitext2 --w_rtn --w_clip \
    #     > "${OUTPUT_DIR}/log_W${W_BIT}A${A_BIT}_No_Rotations.txt" 2>&1

done

echo "All scheduled bit-width experiments are finished."