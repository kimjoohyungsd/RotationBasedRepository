
export MASTER_PORT=$((12000 + $RANDOM % 20000))

python ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--save_safetensors False \
--k_groupsize 128 \
--v_groupsize 128 \
--w_groupsize -1 \
--a_groupsize -1 \
--draw \
--act_check \
--weight_check \
--rotate \
--online_r2 \
# --offline \
# --diagonal \
# --diagonal_size 128 \
# --smooth_quant \
# --alpha 0.6 \
# --attention \










# --a_asym \
