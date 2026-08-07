# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
import os
from pathlib import Path

import random
from typing import Optional
# import wandb

import numpy as np
import torch
from fast_hadamard_transform import hadamard_transform
from torch.distributed.fsdp import (
    FullStateDictConfig,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as PT_FSDP,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def pt_fsdp_state_dict(model: torch.nn.Module):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with PT_FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        return model.state_dict()


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(ctx, u):
        return hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):
        return hadamard_transform(grad)


def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, "groupsize should be greater than 1!"

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f"(Act.) Groupsiz = Down_proj Groupsize: {groupsize}")
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert (
        groupsize * group_num == model.config.hidden_size
    ), "Invalid groupsize for llama!"

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert (
        down_proj_groupsize * group_num == model.config.intermediate_size
    ), "Invalid groupsize for down_proj!"
    logging.info(
        f"(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}"
    )
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())

    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect

    caller_name = ""
    try:
        caller_name = f" (from {inspect.stack()[1].function})"
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(
            torch.cuda.memory_reserved(device=i)
            for i in range(torch.cuda.device_count())
        )

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name: Optional[str],outpath:str=None) -> logging.Logger:
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)
    logger.propagate = False  # 자식의 로그를 부모(Root)에게 전달하지 않음
    
    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add the console handler to the logger
    if outpath:
        log_path = Path(outpath)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(outpath,mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0

# Wandb 관련 값들 작업들
def setup_wandb(input_model,args):
    """Initialize wandb with groups and tags for better organization"""
    if not args.wandb:
        return None
    
    import wandb
    wandb.login()
    
    model_name = input_model.split("/")[-1]

    config = {
        "model": model_name,
        "model_path": input_model,
        "tasks": args.tasks,
        "w_bits": args.w_bits,
        "a_bits": args.a_bits,
        "k_bits": args.k_bits,
        "v_bits": getattr(args, "v_bits", None),
        "w_groupsize": args.w_groupsize,
        "a_groupsize": args.a_groupsize,
        "k_groupsize": args.k_groupsize,
        "w_rtn": getattr(args, "w_rtn", None),
        "w_clip": getattr(args, "w_clip", None),
        "per_column": getattr(args, "per_column", None),
        "smooth_quant": bool(getattr(args, "smooth_quant", False)),
        "rotate": bool(getattr(args, "rotate", False)),
        "batch_size": args.lm_eval_batch_size,
        "seed": args.seed,
    }

    # Build tags
    tags = [
        f"W{args.w_bits}",
        f"A{args.a_bits}",
        f"KV{args.k_bits}",
        model_name,
    ]

    # Build group name (같은 설정의 run들을 그룹화)
    group_name = f"{model_name}-W{args.w_bits}A{args.a_bits}KV{args.k_bits}"
    group_name += f"-Wg{args.w_groupsize}Ag{args.a_groupsize}KVg{args.k_groupsize}"

    if getattr(args, "smooth_quant", False):
        config["smooth_alpha"] = args.alpha
        tags.append("smooth-quant")
        group_name += f"-smooth{args.alpha}"

    if getattr(args, "rotate", False):
        config["rotate_mode"] = getattr(args, "rotate_mode", None)
        config["optimized_rotation_path"] = getattr(args, "optimized_rotation_path", None)
        config["diagonal"] = getattr(args, "diagonal", False)
        config["diagonal_size"] = args.diagonal_size if getattr(args, "diagonal", False) else None
        config["offline"] = getattr(args, "offline", False)
        config['dynamic_residual_scaling'] = getattr(args,'dynamic_residual_scaling',False)
        tags.append('dynamic_residual_scaling')
        tags.append("rotation")
        if getattr(args,'respinquant',False):
            tags.append('respinquant')
        elif getattr(args,'lierespinquant',False):
            tags.append('lierespinquant')
        else:
            tags.append("spinquant" if getattr(args, "optimized_rotation_path", None) else "hadamard")
        group_name += "-rotate"

        if getattr(args, "diagonal", False):
            tags.append(f"diagonal-{args.diagonal_size}")
            group_name += f"-diagonal{args.diagonal_size}"
    else:
        tags.append("no-rotation")

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_id,
        name=group_name,
        group=group_name,
        tags=tags,
        config=config,
    )
    return run


def log_quantization_info(args, log):
    """Log quantization configuration"""
    log.info(f"Quantization bits - W: {args.w_bits}, A: {args.a_bits}, KV: {args.k_bits}")
    
    if args.w_groupsize != -1:
        log.info(f"Quantization group size - W: {args.w_groupsize}, A: {args.a_groupsize}")
    else:
        log.info("Quantization group size - W: per-channel, A: per-token")
    
    if hasattr(args, 'diagonal') and args.diagonal:
        log.info(f"Diagonal size: {args.diagonal_size}")
    
    if args.k_groupsize != -1:
        log.info(f"Quantization group size KV: {args.k_groupsize}")
    else:
        log.info("Quantization group size KV: per-head")
    
    if args.rotate:
        log.info("Rotation available")
        if args.optimized_rotation_path is not None:
            log.info(f"Rotation repository: {args.optimized_rotation_path}")
    
    if args.w_rtn:
        log.info("Using Round-to-nearest method for weight quantization")
    elif args.w_bits < 16:
        log.info("Using GPTQ method for weight quantization")


def log_lm_eval_results(results, args, wandb_run=None):
    """Process and log LM eval results"""
    try:
        metric_vals = {
            task: round(result.get('acc_norm,none', result.get('acc,none', 0)), 4) 
            for task, result in results.items()
        }
        
        # Calculate average
        if metric_vals:
            metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
        
        # Print results
        print("\n" + "="*50)
        print("LM Evaluation Results:")
        print("="*50)
        for task, acc in metric_vals.items():
            print(f"{task:20s}: {acc:.4f}")
        print("="*50 + "\n")
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log(metric_vals)
        
        return metric_vals
        
    except Exception as e:
        error_msg = f"Error processing evaluation results: {e}"
        print(error_msg)
        if wandb_run is not None:
            wandb_run.log({"error": error_msg})
        return None