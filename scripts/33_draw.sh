CUDA_VISIBLE_DEVICES=0,1 python  ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 16 \
--w_rtn \
--w_clip \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--k_asym \
--v_asym \
--a_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--wikitext2 \
--distribute \
--draw \
--act_check \
--wandb \
--wandb_project "rotation-based-evaluation" \
--wandb_id "jhk971114" \
--dynamic_residual_scaling \
# --weight_check \
# --rotate \
# --wikitext2 \
# --distribute \

CUDA_VISIBLE_DEVICES=0,1 python  ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits 16 \
--w_rtn \
--w_clip \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--k_asym \
--v_asym \
--a_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--wikitext2 \
--distribute \
--draw \
--act_check \
--wandb \
--wandb_project "rotation-based-evaluation" \
--wandb_id "jhk971114" \
# --dynamic_residual_scaling \