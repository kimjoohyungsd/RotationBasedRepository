import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast, pipeline
import transformers

# import lm_eval
# from lm_eval import evaluator, utils
# from lm_eval.api.registry import ALL_TASKS
# import lm_eval.tasks 
# from lm_eval.utils import setup_logging 
# from zeroShot.model import SpinquantLMWrapper

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq

import os
# from datasets import load_dataset




def train() -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=100)) # initializes the default distributed process group and Communication backend: NCCL 
    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)
    if ptq_args.wandb:
        import wandb
        wandb.login()
        wandb.init(project=ptq_args.wandb_project, entity=ptq_args.wandb_id)
        wandb.config.update(ptq_args)
    # local_rank = utils.get_local_rank() # 

    # log.info("the rank is {}".format(local_rank)) # 두번 log
    # torch.distributed.barrier() # OS에서 Barrier 설정과 동일 

    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model.cuda() # 모델을 GPU로 옮긴다

# Use a dictionary to log messages as key-value pairs
    if (ptq_args.rotate):
        if ptq_args.wandb:
            wandb.log({"rotation_status": "Rotation available"})
        if ptq_args.optimized_rotation_path is not None:
            if ptq_args.wandb:
                wandb.log({"rotation_path_info": f"Rotation_repository: {ptq_args.optimized_rotation_path}"})

    if (ptq_args.w_rtn):
        if ptq_args.wandb:
            wandb.log({"quantizaton_method": "During Weight Quantization use basic Round-to-nearest method"})
    else:
        if ptq_args.w_bits < 16:
            if ptq_args.wandb:
                wandb.log({"quantization_method": "Use GPTQ method in Weight Quantization"})

    if ptq_args.wandb:
        wandb.log({"quantization_bits": f"W: {ptq_args.w_bits}, A: {ptq_args.a_bits}, KV: {ptq_args.k_bits}"})

    if ptq_args.w_groupsize != -1:
        if ptq_args.wandb:
            wandb.log({"quantization_group_size": f"W: {ptq_args.w_groupsize}, A: {ptq_args.a_groupsize}"})
    else:
        if ptq_args.wandb:
            wandb.log({"quantization_group_size": "W: per-channel, A: per-token"})

    if ptq_args.k_groupsize != -1:
        if ptq_args.wandb:
            wandb.log({"quantization_group_size_kv": f"{ptq_args.k_groupsize}"})
    else:
        if ptq_args.wandb:
            wandb.log({"quantization_group_size_kv": "per-head"})

    model = ptq_model(ptq_args, model, model_args) # 
    model.seqlen = training_args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )

    if not ptq_args.lm_eval:
        log.info("Skipping LM_eval task")
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        # try:
        #     results = lm_eval.simple_evaluate(
        #         model=wrapped_model,
        #         tasks=ptq_args.lm_eval_dat,
        #         num_fewshot=0,
        #         batch_size=8,
        #     )
        #     summary_metrics = results.get("results", {})
        #     formatted_metrics = "\n".join(f"{task}: {metric_dict}" for task, metric_dict in summary_metrics.items())
        #     log.info("Evaluation Metrics Summary:\n{}".format(formatted_metrics))
        #     print("Zero-shot Evaluation Results:")
        #     # print(results) # 주석 해제하여 전체 결과 출력 가능
        # except Exception as e:
        #     log.error(f"Error during zero-shot evaluation with lm_eval harness: {e}")

    # tokenizer = transformers.AutoTokenizer.from_pretrained(ptq_args.model, use_fast=False)
        model.cuda()
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=ptq_args.lm_eval_batch_size)

        # task_names = lm_eval_utils.pattern_match(ptq_args.tasks, ALL_TASKS)
        try:
            results = lm_eval.simple_evaluate(hflm, tasks=ptq_args.tasks, batch_size=ptq_args.lm_eval_batch_size)['results']

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}
            metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
            print(metric_vals)
        except Exception as e:
            wandb.log(f"Error during zero-shot evaluation with lm_eval harness: {e}")
        if ptq_args.wandb:
            wandb.log(metric_vals)



    

if __name__ == "__main__":
    train()
