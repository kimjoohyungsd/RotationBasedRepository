# ReSpinQuant PTQ evaluation.
#   $1 = input model (e.g. meta-llama/Llama-2-7b-hf)
#   $2 = w_bits, $3 = a_bits, $4 = k/v_bits
#   $5 = path to trained R.bin (from optimize_rotation.py --respinquant)
#
# Fuses per-layer R1/R2 into weights offline and applies the rank-r residual
# subspace correction. Requires --rotate + --respinquant + the trained R.bin.
python ptq.py \
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
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--respinquant \
--residual_rank 32 \
--optimized_rotation_path $5 \
--wikitext2
# add --lm_eval --tasks "piqa,arc_easy,..." for zero-shot accuracy
