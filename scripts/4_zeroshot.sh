python zeroshot.py \
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
--a_asym \
--k_asym \
--v_asym \
--lm_eval \
--lm_eval_batch_size 4 \
--tasks "piqa" \
--w_groupsize 64 \
--a_groupsize 64 \
--k_groupsize 128 \
--v_groupsize 128 \
--wandb \
--wandb_project "rotation-based-evaluation" \
--wandb_id "jhk971114" \
# --w_rtn \
# --rotate \




# --optimized_rotation_path  /home/jhkcool97/Rotation_repository/Matrixes/LLAMA-2-7B/R1644.bin \













# --lm_eval_dat "arc_challenge" \

# --save_qmodel_path $5 \
# --eval_out_path /home/jhkcool97/SpinQuant/Results/Zeroshot/Arc_challenge/W4A4KV4/W,A:32,KV:head/RTN/Result.txt \


# --optimized_rotation_path "your_path/R.bin" \
# --output
