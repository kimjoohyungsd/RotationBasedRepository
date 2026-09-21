#!/bin/bash
# MXFP4 BRQ-GPTQ (block-diagonal random-Hadamard rotation + W4A4 MXFP4, GPTQ weights) PPL evaluation.
# Same as scripts/MXFP4/BRQ/llama/Parallel/2_eval_ptq.sh but the weights are quantized by GPTQ
# (on the MXFP4 grid) instead of RTN, i.e. --w_rtn is dropped.
# Runs the three Llama models concurrently, each model_parallel-distributed over its own GPUs:
#   Llama-2-7b   -> GPU 0,1
#   Llama-2-13b  -> GPU 2,3,4
#   Llama-3-8B   -> GPU 5,6
# Every model uses the identical recipe (MXFP4 block 32 for W and A, block size-32 random-Hadamard rotation).
# Logs: logs/MXFP4/BRQ-GPTQ/<model_name>/
#
# Env overrides: PY (python), ONLY (model-id regex), NSAMPLES, PERCDAMP, ACT_ORDER=1 (GPTQ act-order, static MX block scales)

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
    "meta-llama/Meta-Llama-3-8B|5,6"
)

# ONLY="Llama-2" (regex on the HF model id) restricts the run to the matching JOBS
if [ -n "$ONLY" ]; then
    FILTERED=()
    for J in "${JOBS[@]}"; do [[ "${J%%|*}" =~ $ONLY ]] && FILTERED+=("$J"); done
    JOBS=("${FILTERED[@]}")
fi

MX_BLOCK=32   # MXFP4 group size == rotation block size (same as BRQ)

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

    # Block-diagonal random Hadamard rotation
    --rotate
    --rotate_mode hadamard
    --diagonal
    --diagonal_size "$MX_BLOCK"

    # KV cache left at 16-bit
    --k_bits 16
    --v_bits 16

    # Evaluation, model spread over the GPUs made visible to each job
    --wikitext2
    --distribute
)

PIDS=()
NAMES=()

for JOB in "${JOBS[@]}"; do
    MODEL_PATH="${JOB%%|*}"
    CUDA_DEVICES="${JOB##*|}"
    MODEL_NAME=$(basename "$MODEL_PATH")
    OUTPUT_DIR="${REPO}/logs/MXFP4/BRQ-GPTQ/${MODEL_NAME}"
    mkdir -p "$OUTPUT_DIR"

    echo ">>> ${MODEL_NAME} on GPU ${CUDA_DEVICES} -> ${OUTPUT_DIR}"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PY ptq.py \
        --input_model "$MODEL_PATH" \
        "${COMMON_ARGS[@]}" \
        "${GPTQ_ARGS[@]}" \
        --eval_out_path "${OUTPUT_DIR}/log_W4A4_MXFP4_BRQ_GPTQ.txt" \
        > "${OUTPUT_DIR}/stdout.txt" 2>&1 &
    PIDS+=($!)
    NAMES+=("$MODEL_NAME")
done

STATUS=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[done]   ${NAMES[$i]}"
    else
        echo "[FAILED] ${NAMES[$i]} (see logs/MXFP4/BRQ-GPTQ/${NAMES[$i]}/stdout.txt)"
        STATUS=1
    fi
done

exit $STATUS
