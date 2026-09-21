#!/bin/bash
# SmoothQuant calibration: per-channel activation scales/shifts for the three Llama models.
# Runs the three models concurrently, each model_parallel-distributed over its own GPUs (see JOBS),
# so models that do not fit on one RTX 3090 (e.g. Llama-2-13b) can be calibrated too.
# Outputs: act_scales/<model_name>.pt, act_shifts/<model_name>.pt (read by 2_eval_ptq.sh via --smooth_quant)
# Logs: logs/MXFP4/SmoothQuant/act_scales/<model_name>/stdout.txt

REPO="/home/jhkcool97/RotationBasedRepository"
cd "$REPO" || exit 1

cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$
    exit 1
}
trap cleanup SIGINT

# "<HF model id>|<CUDA_VISIBLE_DEVICES>"
JOBS=(
    "meta-llama/Llama-2-7b-hf|0,1"
    "meta-llama/Llama-2-13b-hf|2,3,4"
    "meta-llama/Llama-3.1-8B|5,6"
)

COMMON_ARGS=(
    --model_max_length 2048
    --fp16 False
    --bf16 True

    # Calibration data (wikitext2 train split)
    --nsamples 128

    # Model spread over the GPUs made visible to each job
    --distribute
)

PIDS=()
NAMES=()
LOG_DIRS=()

for JOB in "${JOBS[@]}"; do
    MODEL_PATH="${JOB%%|*}"
    CUDA_DEVICES="${JOB##*|}"
    MODEL_NAME=$(basename "$MODEL_PATH")

    echo ">>> ${MODEL_NAME} on GPU ${CUDA_DEVICES} -> ${OUTPUT_DIR}"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python generate_act_scale.py \
        --input_model "$MODEL_PATH" \
        "${COMMON_ARGS[@]}" \
        2>&1 &
    PIDS+=($!)
    NAMES+=("$MODEL_NAME")
    LOG_DIRS+=("$OUTPUT_DIR")
done

STATUS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[done]   ${NAMES[$i]}"
    else
        echo "[FAILED] ${NAMES[$i]} (see ${LOG_DIRS[$i]}/stdout.txt)"
        STATUS=1
    fi
done

exit $STATUS
