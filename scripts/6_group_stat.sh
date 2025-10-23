
export MASTER_PORT=$((12000 + $RANDOM % 20000))

python group_stat.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--save_safetensors False \
--w_bits 16 \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize -1 \
--a_groupsize -1 \
--k_asym \
--v_asym \
--w_rtn \
--wikitext2 \
--w_clip \
--stats_path '/home/jhkcool97/SpinQuant/stats' \
# --rotate \
# --offline \
# --diagonal \
# --diagonal_size 128 \
# --smooth_quant \
# --alpha 0.6 \
# --attention \










# --a_asym \
