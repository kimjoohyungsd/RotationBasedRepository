# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
# import torch.distributed as dist
from transformers import LlamaTokenizerFast,PreTrainedTokenizerFast # LlamaForCausalLM, pipeline
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

from datasets import load_dataset




def train() -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=100)) # initializes the default distributed process group and Communication backend: NCCL 
    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)

    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 or config.torch_dtype==torch.bfloat16 else torch.float16
    model = LlamaForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda() # 모델을 GPU로 옮긴다

    if (ptq_args.rotate):
        log.info("Rotation applied")
        if ptq_args.optimized_rotation_path is not None:
            log.info("Trained Rotation Matrix applied")

    log.info("Quantization bits W: {},A: {}, KV: {}".format(ptq_args.w_bits,ptq_args.a_bits,ptq_args.k_bits))

    a_groupsize = ptq_args.a_groupsize if ptq_args.a_groupsize != -1 else "per-token"
    w_groupsize = ptq_args.w_groupsize if ptq_args.w_groupsize != -1 else "per-channel"
    kv_groupsize = ptq_args.k_groupsize if ptq_args.k_groupsize != -1 else 128
    log.info("Quantization group size W: {}, A: {}, KV: {}".format(a_groupsize,w_groupsize,kv_groupsize))

    model = ptq_model(ptq_args, model, log, model_args) # 
    model.seqlen = training_args.model_max_length

    if 'Llama-3' in model_args.input_model:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
        )
    else:
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
    log.info("Complete tokenizer loading...")
    

    if ptq_args.wikitext2:
        model.config.use_cache = False
        testloader = data_utils.get_wikitext2( 
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))
        # dist.barrier()

    if not ptq_args.lm_eval:
        log.info("Skipping LM_eval task")

    # Setup wandb (only once)
    
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM
        wandb_run = utils.setup_wandb(model_args.input_model,ptq_args) if local_rank == 0 else None

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

        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    train()
