# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

from datasets import load_dataset




def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=100)) # initializes the default distributed process group and Communication backend: NCCL 
    model_args, training_args, ptq_args = process_args_ptq()
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)
    local_rank = utils.get_local_rank() # 

    log.info("the rank is {}".format(local_rank)) # 두번 log
    torch.distributed.barrier() # OS에서 Barrier 설정과 동일 

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
    # model=transformers.LlamaForCausalLM.from_pretrained(
    #            pretrained_model_name_or_path=model_args.input_model,
    #     config=config,
    #     torch_dtype=dtype,
    #     token=model_args.access_token,
    # )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.cuda() # 모델을 GPU로 옮긴다

    if (ptq_args.rotate):
        log.info("Rotation available")
        if ptq_args.optimized_rotation_path is not None:
            log.info("Rotation_repository{}".format(ptq_args.optimized_rotation_path))

    if (ptq_args.w_rtn):
        log.info("During Weight Quantization use basic Round-to-nearest method")
    else:
        if ptq_args.w_bits<16:
            log.info("Use GPTQ method in Weight Quantization")

    log.info("Quantization bits W: {},A: {}, KV: {}".format(ptq_args.w_bits,ptq_args.a_bits,ptq_args.k_bits))

    if ptq_args.w_groupsize != -1:
        log.info("Quantization group size W: {},A:{}".format(ptq_args.w_groupsize,ptq_args.a_groupsize))
    else:
        log.info("Quantization group size W: per-channel,A:per-token")
    
    if ptq_args.k_groupsize != -1:
        log.info("Quantization group size KV: {}".format(ptq_args.k_groupsize))
    else:
        log.info("Quantization group size KV: per-head")

    model = ptq_model(ptq_args, model, model_args) # 
    model.seqlen = training_args.model_max_length
    if local_rank == 0:
        log.info("Model PTQ completed {}".format(model))
        log.info("Start to load tokenizer...")
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
        testloader = data_utils.get_wikitext2( #
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
        log.info("wiki2 ppl is: {}".format(dataset_ppl))
        dist.barrier()

    
    # if local_rank ==  0 and ptq_args.lm_eval_dat is not None:
    #     model.config.use_cache = True
    #     log.info("Starting zero-shot evaluation with lm_eval harness...")
    #     model.cuda()
    #     wrapped_model=SpinquantLMWrapper(pretrained=model,tokenizer=tokenizer,max_length=model.seqlen)
    #     try:
    #         results = lm_eval.simple_evaluate(
    #             model=wrapped_model,
    #             tasks=ptq_args.lm_eval_dat,
    #             num_fewshot=0,
    #             batch_size=8,
    #         )
    #         summary_metrics = results.get("results", {})
    #         formatted_metrics = "\n".join(f"{task}: {metric_dict}" for task, metric_dict in summary_metrics.items())
    #         log.info("Evaluation Metrics Summary:\n{}".format(formatted_metrics))
    #         print("Zero-shot Evaluation Results:")
    #         # print(results) # 주석 해제하여 전체 결과 출력 가능
    #     except Exception as e:
    #         log.error(f"Error during zero-shot evaluation with lm_eval harness: {e}")
    #         # print(results)
        #2: Test for Arc-E dataset

        
        # pipe= pipeline
        # dataset=load_dataset("arc-e")

if __name__ == "__main__":
    train()
