#!/bin/bash
# MXFP4 SmoothQuant-GPTQ (alpha grid 0.3..0.8, GPTQ weights, no rotation) PPL evaluation.
# Qwen3-8B / Qwen3-14B, W4A4 MXFP4, KV cache 16-bit, WikiText-2 PPL.
# (Qwen/Qwen3-7B does not exist on the HF Hub; Qwen3-8B is the model between 4B and 14B.)
# Derived from scripts/MXFP4/SmoothQuant/qwen3/2_eval_ptq.sh with --w_rtn dropped (=> GPTQ weights).
# Models run one after another, each model_parallel-distributed (--distribute) over CUDA_DEVICES.
# Logs: logs/MXFP4/SmoothQuant_GPTQ/<model_name>/alpha_<alpha>/
#
# !! Qwen3 needs the packages of requirement_qwen3.txt (transformers==4.57.0, accelerate==1.1.0,
# !! lm-eval==0.4.5, peft==0.13.2) installed in the python used here (override with PY=...).
#
# Env overrides: PY, CUDA_DEVICES, ALPHAS="0.5 0.6", NSAMPLES, PERCDAMP, ACT_ORDER=1

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

MODELS=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B")
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

MX_BLOCK=32   # MXFP4 group size

# SmoothQuant migration strength grid (same as SmoothQuant/RTN)
read -r -a ALPHAS <<< "${ALPHAS:-0.3 0.4 0.5 0.6 0.7 0.8}"

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

    # KV cache left at 16-bit
    --k_bits 16
    --v_bits 16

    # Evaluation, model spread over the GPUs made visible to the job
    --wikitext2
    --distribute
)

STATUS=0

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL_PATH")

    # SmoothQuant needs per-channel activation abs-max statistics: act_scales/<model>.pt
    if [ ! -f "${REPO}/act_scales/${MODEL_NAME}.pt" ]; then
        echo ">>> generating act_scales for ${MODEL_NAME}"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PY generate_act_scale_qwen3.py \
            --input_model "$MODEL_PATH" || { echo "[FAILED] act_scales ${MODEL_NAME}"; STATUS=1; continue; }
    fi

    for ALPHA in "${ALPHAS[@]}"; do
        OUTPUT_DIR="${REPO}/logs/MXFP4/SmoothQuant_GPTQ/${MODEL_NAME}/alpha_${ALPHA}"
        mkdir -p "$OUTPUT_DIR"

        echo ">>> ${MODEL_NAME} alpha=${ALPHA} on GPU ${CUDA_DEVICES} -> ${OUTPUT_DIR}"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES $PY ptq_qwen3.py \
            --input_model "$MODEL_PATH" \
            "${COMMON_ARGS[@]}" \
            "${GPTQ_ARGS[@]}" \
            --smooth_quant \
            --alpha "$ALPHA" \
            --eval_out_path "${OUTPUT_DIR}/log_W4A4_MXFP4_SmoothQuant_GPTQ.txt" \
            > "${OUTPUT_DIR}/stdout.txt" 2>&1 \
            && echo "[done]   ${MODEL_NAME} alpha=${ALPHA}" \
            || { echo "[FAILED] ${MODEL_NAME} alpha=${ALPHA} (see ${OUTPUT_DIR}/stdout.txt)"; STATUS=1; }
    done
done

exit $STATUS
