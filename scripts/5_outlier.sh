export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python outlier_check.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 True \
--bf16 False \
--save_safetensors False \
--distribution_dir /home/jhkcool97/Model_distribution/Llama-3.1-8b \
--capture_layer_io \
--weight_check \
--draw \
# --rotate \
# --smooth_quant \
# --alpha 0.60 \
# --attention \
# --diagonal \
# --diagonal_size 128 \




















# --rotate \
# --optimized_rotation_path /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R1644.bin \
# --w_bits $2 \
# --a_bits $3 \
# --k_bits $4 \
# --v_bits $4 \
# --lm_eval \
# --lm_eval_batch_size 4 \
# --tasks "arc_challenge" \
# --w_groupsize 32 \
# --a_groupsize 32 \
# --k_groupsize 128 \
# --v_groupsize 128 \
# --wandb \
# --wandb_project "rotation-based-evaluation" \
# --wandb_id "jhk971114" \
# --w_rtn \
# --w_clip \
# --a_asym \
# --k_asym \
# --v_asym \