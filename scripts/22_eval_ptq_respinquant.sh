# ReSpinQuant PTQ evaluation.
#   $1 = input model (e.g. meta-llama/Llama-2-7b-hf)
#   $2 = w_bits, $3 = a_bits, $4 = k/v_bits
#   $5 = path to trained R.bin (from optimize_rotation.py --respinquant)
#
# Fuses per-layer R1/R2 into weights offline and applies the rank-r residual
# subspace correction. Requires --rotate + --respinquant + the trained R.bin.
CUDA_VISIBLE_DEVICES=2,3 python ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--k_asym \
--v_asym \
--a_asym \
--wikitext2 \
--rotate \
--respinquant \
--residual_rank 32 \
--optimized_rotation_path "/home/jhkcool97/Rotation_repository/Matrixes/LLAMA-3-8B/ReSpinQuant/W:16A:4KV:4/R.bin" \
--deactivate_residual \
--k_groupsize 128 \
--v_groupsize 128 \
--eval_out_path "/home/jhkcool97/RotationBasedRepository/logs/Llama-3.1-8b/ReSpinQuant/w4a4kv4.txt" \
--distribute \
--wandb \
--wandb_project "rotation-based-evaluation" \
--wandb_id "jhk971114" \

# --lm_eval \
# --lm_eval_batch_size 256 \
# --distribute
# add --lm_eval --tasks "piqa,arc_easy,..." for zero-shot accuracy
#
# ── 16-bit sanity checks (run with $2=$3=$4=16 to disable quantization) ──
# 1) EXACT correction  → PPL MUST match the un-rotated baseline (fusion is lossless).
#      --residual_rank 99999    (any r >= hidden_size, or r <= 0, triggers the exact path)
#    If PPL matches baseline: fusion logic is correct, and high PPL at rank 32 is the
#    approximation being too coarse. If it does NOT match: real bug in the fusion.
# 2) Correction OFF    → expected to be HIGH even at 16-bit (per-layer R1/R2 leave a
#    residual-stream basis mismatch). Use only as an A/B reference, not as a pass check.
#      --deactivate_residual
