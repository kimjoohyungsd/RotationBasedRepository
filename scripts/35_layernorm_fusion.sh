#!/bin/bash
cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$ 
    exit 1
}
trap cleanup SIGINT

# MODELS=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-3.1-8B" )
# MODELS=( "meta-llama/Llama-3.1-8B" )
# MODELS=("Qwen/Qwen3-8B" )

MODELS=('Qwen/Qwen3-14B')
COMMON_ARGS=(
    --do_train False
    --do_eval True

    --per_device_eval_batch_size 4
    --model_max_length 2048

    --fp16 False
    --bf16 True
    --save_safetensors False

    # Weight / Activation Quantization
    --w_bits 4
    --a_bits 4
    --w_rtn
    --mxfp4
    --a_asym

    # KV Cache
    --k_bits 16
    --v_bits 16
    --k_asym
    --v_asym
    --k_groupsize 128
    --v_groupsize 128

    # Evaluation
    --wikitext2
    --distribute

    # WandB
    --wandb
    --wandb_project "rotation-based-evaluation"
)

for MODEL_PATH in "${MODELS[@]}"; do

    MODEL_NAME="${MODEL_PATH}"
    if [[ "$MODEL_PATH" == *"Qwen3"* ]] || [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
        PY_SCRIPT="ptq_qwen3.py"
        echo ">>> Qwen3 model detected. Using $PY_SCRIPT"
    else
        PY_SCRIPT="ptq.py"
        echo ">>> Standard model detected. Using $PY_SCRIPT"
    fi

    echo "======================================================"
    echo "Model: ${MODEL_PATH}"
    echo "======================================================"


    echo "[Experiment 1] LayerNormFusion"

    CUDA_VISIBLE_DEVICES=0,1,2,3 python $PY_SCRIPT \
        --input_model "$MODEL_PATH" \
        "${COMMON_ARGS[@]}" \
        --norm_fusion > &
    
    echo "[Experiment 2] X Layer Norm Fusion"

    CUDA_VISIBLE_DEVICES=4,5,6,7 python $PY_SCRIPT \
        --input_model "$MODEL_PATH" \
        "${COMMON_ARGS[@]}" &

done