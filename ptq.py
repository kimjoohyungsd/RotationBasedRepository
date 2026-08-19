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
from transformers import LlamaTokenizerFast,PreTrainedTokenizerFast,AutoTokenizer # LlamaForCausalLM, pipeline
import transformers
import os

# import lm_eval
# from lm_eval import evaluator, utils
# from lm_eval.api.registry import ALL_TASKS
# import lm_eval.tasks 
# from lm_eval.utils import setup_logging 
# from zeroShot.model import SpinquantLMWrapper

from eval_utils.main import ptq_model
from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils.modeling_qwen2 import Qwen2ForCausalLM # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?

from utils import data_utils, eval_utils, utils, draw_utils, parallel_utils
from utils.process_args import process_args_ptq


def distribution_subdir(ptq_args) -> str:
    """Name the distribution-plot directory after *every* transform that ran.

    The transforms are not mutually exclusive -- smoothing, permutation, rotation
    and FPTQuant's Sn can all be applied in the same run -- so the name is a
    ' + '-joined list in pipeline order (see eval_utils.main.ptq_model), e.g.

        Smoothed (a=0.6) + Rotated (LieReSpinQuant) + Sn

    A run with a single transform keeps the name it had before ("Rotated (SpinQuant)",
    "Smoothed"), so previously written figure directories are unaffected.
    """
    parts = []

    if getattr(ptq_args, "smooth_quant", False):
        alpha = getattr(ptq_args, "alpha", None)
        parts.append("Smoothed" if alpha is None else "Smoothed (a={})".format(alpha))

    if getattr(ptq_args, "permute", False):
        mode = getattr(ptq_args, "permute_mode", None)
        parts.append("Permuted" if mode is None else "Permuted ({})".format(mode))

    if getattr(ptq_args, "rotate", False):
        if getattr(ptq_args, "lierespinquant", False):
            flavor = "LieReSpinQuant r={}".format(getattr(ptq_args, "lie_rank", 32))
        elif getattr(ptq_args, "respinquant", False):
            rank = getattr(ptq_args, "residual_rank", 32)
            flavor = "ReSpinQuant r={}".format(rank)
        elif getattr(ptq_args, "optimized_rotation_path", None):
            flavor = "SpinQuant"
        else:
            flavor = getattr(ptq_args, "rotate_mode", "hadamard")
        parts.append("Rotated ({})".format(flavor))

    if getattr(ptq_args, "dynamic_residual_scaling", False):
        parts.append("Sn")

    return " + ".join(parts) if parts else "Baseline"


from datasets import load_dataset




def train() -> None:
    # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=100)) # initializes the default distributed process group and Communication backend: NCCL 
    model_args, training_args, ptq_args = process_args_ptq()

    # ptq_args.eval_out_path = os.path.join(ptq_args.eval_out_path,f"{}")
    log: Logger = utils.get_logger("spinquant",ptq_args.eval_out_path)

    # Setup wandb (only once, before anything heavy runs so the config is recorded even if eval crashes)
    wandb_run = utils.setup_wandb(model_args.input_model, ptq_args)
    if wandb_run is not None:
        log.info("wandb run: {} ({})".format(wandb_run.name, wandb_run.url))

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 or config.torch_dtype==torch.bfloat16 else torch.float16
    if ptq_args.draw:
        dtype = torch.float16

    device_map = "auto" if ptq_args.distribute else None
    n_gpus = torch.cuda.device_count()

    # 0번 GPU에는 파라미터를 적게(예: 16GB), 나머지는 넉넉히(예: 22GB) 할당
    # 70B 모델은 전체 약 140GB(FP16) / 35GB(W4)이므로 이에 맞춰 분배
    max_memory = {}  # 빈 딕셔너리로 초기화
    if ptq_args.distribute:
        n_gpus = torch.cuda.device_count()
        # 0번 GPU는 Activation 공간 확보를 위해 적게 할당
        max_memory[0] = "10GiB" 
        for i in range(1, n_gpus):
            # 나머지 GPU는 모델 파라미터를 담기 위해 더 넉넉히 할당 (예: 24GB 카드 기준)
            max_memory[i] = "10GiB" 
    else:
        max_memory = None # 분산 모드가 아닐 때는 None 전달

    model_args.net= model_args.input_model.split('/')[-1]
    if 'Llama' in model_args.net:
        model = LlamaForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
            device_map=device_map,
            max_memory=max_memory
        )
    elif 'Qwen' in model_args.net:
        model = Qwen2ForCausalLM.from_pretrained( # 왜 Eval_utils에서 modeling_llama 파일을 overwrite 했을까?
            pretrained_model_name_or_path=model_args.input_model,
            config=config,
            torch_dtype=dtype,
            token=model_args.access_token,
            device_map=device_map,
            max_memory=max_memory
        )
    # elif 'Qwen3' in model_args.net:
    #     model = Qwen3ForCausalLM.from_pretrained(
    #         pretrained_model_name_or_path=model_args.input_model,
    #         config=config,
    #         torch_dtype=dtype,
    #         token=model_args.access_token,
    #         device_map=device_map
    #     )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    # --gptq_cpu_offload keeps the model on CPU: gptq_fwrd_distribute streams one layer
    # at a time to a GPU, so the full 70B model must NOT be pinned to a single GPU here.
    if not ptq_args.distribute and not getattr(ptq_args, "gptq_cpu_offload", False):
        model.cuda() # 모델을 GPU로 옮긴다
    elif getattr(ptq_args, "gptq_cpu_offload", False):
        assert not ptq_args.distribute, "--gptq_cpu_offload is incompatible with --distribute (device_map)."
        log.info("gptq_cpu_offload: model kept on CPU; layers streamed to GPU during GPTQ.")

    if (ptq_args.rotate):
        log.info("Rotation applied")
        if ptq_args.optimized_rotation_path is not None:
            log.info("Trained Rotation Matrix applied")

    log.info("Quantization bits W: {},A: {}, KV: {}".format(ptq_args.w_bits,ptq_args.a_bits,ptq_args.k_bits))

    a_groupsize = ptq_args.a_groupsize if ptq_args.a_groupsize != -1 else "per-token"
    w_groupsize = ptq_args.w_groupsize if ptq_args.w_groupsize != -1 else "per-channel"
    kv_groupsize = ptq_args.k_groupsize if ptq_args.k_groupsize != -1 else 128
    log.info("Quantization group size W: {}, A: {}, KV: {}".format(a_groupsize,w_groupsize,kv_groupsize))
    
    if ptq_args.per_column:
        log.info("Quantization is done on column wise manner")
        
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
    elif 'Qwen' in model_args.input_model:
        tokenizer = AutoTokenizer.from_pretrained(
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

    model = ptq_model(ptq_args, model, log, tokenizer, model_args) # 
    model.seqlen = training_args.model_max_length

    # if ptq_args.distribute and not ptq_args.w_rtn:

    
    results = {}
    if ptq_args.wikitext2 or ptq_args.draw:
        model.config.use_cache = False
        testloader = data_utils.get_wikitext2( 
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

        if ptq_args.wikitext2:
            dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
            results['wiki2_ppl'] = dataset_ppl
            if wandb_run is not None:
                wandb_run.log({"wiki2_ppl": dataset_ppl})
                wandb_run.summary["wiki2_ppl"] = dataset_ppl

        if ptq_args.draw:
            if ptq_args.distribution_dir is None:
                ptq_args.distribution_dir = os.path.join("figures", model_args.net)

            os.makedirs(ptq_args.distribution_dir, exist_ok=True)

            save_path = os.path.join(
                ptq_args.distribution_dir, distribution_subdir(ptq_args))

            if ptq_args.weight_check:

                weight_path = os.path.join(save_path,"Weight")
                os.makedirs(weight_path,exist_ok=True)

                # draw_utils에서 모델의 분포 및 3차원 그래프를 그리는 것을 진행하지
                draw_utils.draw_weight(model,weight_path,ptq_args)

            if ptq_args.act_check:
                act_path = os.path.join(save_path,"Act")
                draw_utils.draw_activations(model,act_path,ptq_args,testloader)


        # dist.barrier()


    if ptq_args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        if not ptq_args.distribute:
            model.cuda() # 모델을 GPU로 옮긴다
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto",max_batch_size=ptq_args.lm_eval_batch_size)

        task_names = ptq_args.tasks

        metric_vals = {}
        for task_name in task_names:
            hflm.batch_sizes = {}   # 캐시 리셋
            log.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name])['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            metric_vals[task_name] = acc
            log.info(f"acc: {acc}%")
            # task 하나 끝날 때마다 올려서 중간에 죽어도 결과가 남도록
            if wandb_run is not None:
                wandb_run.log({task_name: acc})
                wandb_run.summary[task_name] = acc

        if metric_vals:
            metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
            results['acc_avg'] = metric_vals['acc_avg']
        log.info(metric_vals)
        log.info(results)
        if wandb_run is not None and metric_vals:
            wandb_run.log({"acc_avg": metric_vals['acc_avg']})
            wandb_run.summary["acc_avg"] = metric_vals['acc_avg']
    else:
        log.info("Skipping LM_eval task")

    if wandb_run is not None:
        import wandb
        # runs 테이블에 한 번에 보이도록 최종 결과를 summary로 정리
        wandb_run.summary.update(results)
        # 결과 테이블(여러 run 비교용)
        if results:
            table = wandb.Table(
                columns=["metric", "value"],
                data=[[k, v] for k, v in results.items()],
            )
            wandb_run.log({"results": table})
        wandb_run.finish()

if __name__ == "__main__":
    train()
