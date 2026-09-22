#!/bin/bash
# MXFP4 SmoothQuant-GPTQ (SmoothQuant + W4A4 MXFP4, GPTQ weights) PPL evaluation with an alpha sweep.
# Same as scripts/MXFP4/SmoothQuant/llama/parallel/2_eval_ptq.sh but the weights are quantized by GPTQ
# (on the MXFP4 grid) instead of RTN, i.e. --w_rtn is dropped.
# For each alpha in ALPHAS, runs the three Llama models concurrently (each model_parallel-distributed
# over its own GPUs, see JOBS); the next alpha starts once all three models have finished.
# Requires act_scales/<model_name>.pt (run 1_generate_act_scale.sh first; generated automatically if missing).
# Logs: logs/MXFP4/SmoothQuant_GPTQ/<model_name>/alpha_<alpha>/
#
# Env overrides: PY (python), ONLY (model-id regex), ALPHAS="0.5 0.6", NSAMPLES, PERCDAMP, ACT_ORDER=1 (GPTQ act-order)

REPO="/home/jhkcool97/RotationBasedRepository"
PY="${PY:-python}"
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

# ONLY="Llama-2" (regex on the HF model id) restricts the run to the matching JOBS
if [ -n "$ONLY" ]; then
    FILTERED=()
    for J in "${JOBS[@]}"; do [[ "${J%%|*}" =~ $ONLY ]] && FILTERED+=("$J"); done
    JOBS=("${FILTERED[@]}")
fi

MX_BLOCK=32   # MXFP4 group size

# SmoothQuant migration strength candidates (same grid as SmoothQuant/RTN)
read -r -a ALPHAS <<< "${ALPHAS:-0.3 0.4 0.5 0.6 0.7 0.8}"

# GPTQ calibration (wikitext2 train split) / damping
NSAMPLES="${NSAMPLES:-128}"
PERCDAMP="${PERCDAMP:-0.01}"
GPTQ_ARGS=(--nsamples "$NSAMPLES" --percdamp "$PERCDAMP")
[ "${ACT_ORDER:-0}" = "1" ] && GPTQ_ARGS+=(--act_order)

COMMON_ARGS=(
    --do_train False
    --do_eval True

    --per_device_eval_batch_size 4
    --model_max_length 2048

    --fp16 False
    --bf16 True
    --save_safetensors False

    # W4A4 in MXFP4 units. NO --w_rtn: weights are quantized by GPTQ (on the MXFP4 grid)
    --w_bits 4
    --a_bits 4
    --mxfp4
    --mx_block "$MX_BLOCK"
    --a_asym

    # SmoothQuant applied
    --smooth_quant
    --attention

    # KV cache left at 16-bit
    --k_bits 16
    --v_bits 16

    # Evaluation, model spread over the GPUs made visible to each job
    --wikitext2
    --distribute
)

# SmoothQuant needs per-channel activation abs-max statistics: act_scales/<model>.pt
if ! PY="$PY" bash "$(dirname "$0")/1_generate_act_scale.sh"; then
    echo "[FAILED] act_scales generation"
    exit 1
fi

STATUS=0

for ALPHA in "${ALPHAS[@]}"; do
    echo "===== alpha = ${ALPHA} ====="
    PIDS=()
    NAMES=()
    LOG_DIRS=()

    for JOB in "${JOBS[@]}"; do
        MODEL_PATH="${JOB%%|*}"
        CUDA_DEVICES="${JOB##*|}"
        MODEL_NAME=$(basename "$MODEL_PATH")
        OUTPUT_DIR="${REPO}/logs/MXFP4/SmoothQuant_GPTQ/${MODEL_NAME}/alpha_${ALPHA}"
        mkdir -p "$OUTPUT_DIR"

        echo ">>> ${MODEL_NAME} (alpha=${ALPHA}) on GPU ${CUDA_DEVICES} -> ${OUTPUT_DIR}"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PY ptq.py \
            --input_model "$MODEL_PATH" \
            "${COMMON_ARGS[@]}" \
            "${GPTQ_ARGS[@]}" \
            --alpha "$ALPHA" \
            --eval_out_path "${OUTPUT_DIR}/log_W4A4_MXFP4_SmoothQuant_GPTQ.txt" \
            > "${OUTPUT_DIR}/stdout.txt" 2>&1 &
        PIDS+=($!)
        NAMES+=("$MODEL_NAME")
        LOG_DIRS+=("$OUTPUT_DIR")
    done

    # Wait for all models of this alpha before moving on (each model owns its GPUs)
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "[done]   ${NAMES[$i]} (alpha=${ALPHA})"
        else
            echo "[FAILED] ${NAMES[$i]} (alpha=${ALPHA}) (see ${LOG_DIRS[$i]}/stdout.txt)"
            STATUS=1
        fi
    done
done

exit $STATUS
