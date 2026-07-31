# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import random_hadamard_matrix, hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

log: Logger = get_logger("spinquant")


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    torch.cuda.memory._record_memory_history(
       max_entries=100000
   )
    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = prepare_model(ptq_args, model)
    for param in model.parameters():
        param.requires_grad = False

    hidden_size = model.config.hidden_size
    head_dim = hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    # Propagate the training mode to the (shared) config so the modeling forward
    # can pick the SpinQuant vs. ReSpinQuant path at runtime.
    model.config.respinquant = training_args.respinquant

    if training_args.respinquant:
        # ---------- ReSpinQuant: full-size layer-wise residual-stream rotations ----------
        #   layers[i].R1 : attention-input / previous-FFN-output basis (hidden x hidden)
        #   layers[i].R2 : attention-output / FFN-input basis        (hidden x hidden)
        #   model.model.R1_final : last FFN-output basis, un-rotated before lm_head
        #   layers[i].self_attn.R2 : head_dim rotation (paper's R3)
        # Residual-stream rotations (R1, R2, R1_final) MUST share the SAME Hadamard base
        # so that the residual transition T = R1_in^T R2_mid ~= H^T H = I at init (paper Eq. 4).
        # Using independent random_hadamard_matrix() per matrix gives distinct sign diagonals,
        # so T = diag(s1 ⊙ s2) != I and Delta_T = T - I is full-rank, which breaks the rank-r
        # residual subspace approximation. `hadamard_matrix` is the deterministic Walsh-Hadamard
        # (no random sign), so every call returns the identical H -> T = I at init. Cayley then
        # lets each R deviate slightly, keeping Delta_T low-rank as the method requires.
        for i in range(num_layers):
            model.model.layers[i].R1 = RotateModule( # model.model.layers[i].R1 => Attention의 Input을 rotation
                hadamard_matrix(hidden_size, "cuda")
            )
            model.model.layers[i].R2 = RotateModule( # model.model.layers[i].R2 => MLP의 Input을 rotation
                hadamard_matrix(hidden_size, "cuda")
            )
            model.model.layers[i].self_attn.R2 = RotateModule( # R3 (head_dim): fused inside attention, never crosses a residual add -> independent random is fine
                random_hadamard_matrix(head_dim, "cuda")
            )
        model.model.R1_final = RotateModule( # 마지막 DecoderLayer의 MLP Block의 Output 축과 Lm head 단위의 Rotation
            hadamard_matrix(hidden_size, "cuda")
        )
    else:
        # ---------- SpinQuant: single global R1 + per-layer head rotation R2 ----------
        model.R1 = RotateModule(random_hadamard_matrix(hidden_size, "cuda"))
        for i in range(num_layers):
            model.model.layers[i].self_attn.R2 = RotateModule(
                random_hadamard_matrix(head_dim, "cuda")
            )
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
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
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    if training_args.respinquant:
        trainable_parameters = []
        for i in range(num_layers):
            trainable_parameters.append(model.model.layers[i].R1.weight)
            trainable_parameters.append(model.model.layers[i].R2.weight)
            trainable_parameters.append(model.model.layers[i].self_attn.R2.weight)
        trainable_parameters.append(model.model.R1_final.weight)
    else:
        trainable_parameters = [model.R1.weight] + [
            model.model.layers[i].self_attn.R2.weight
            for i in range(num_layers)
        ]
    model.seqlen = training_args.model_max_length

    # Disk-safety: unless --save_checkpoints is passed, disable HF Trainer checkpointing
    # so a full-disk error mid-run cannot waste the training. Each Trainer checkpoint
    # dumps the full model + optimizer state (tens of GB); we don't need them here since
    # the trained rotations are saved separately to output_rotation_path/R.bin below.
    if not getattr(ptq_args, "save_checkpoints", False):
        training_args.save_strategy = "no"
        if local_rank == 0:
            log.info("Trainer checkpointing disabled (save_strategy='no'); only R.bin will be written. "
                     "Pass --save_checkpoints to keep intermediate checkpoints.")

    optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate, stiefel=True)
    MyTrainer = Trainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    torch.distributed.barrier()

    try:
        trainer.train()
    except Exception as e:
        log.error(f"Trainer Failed with exception {e}")
        try:
            file_prefix = os.path.join(training_args.output_dir, "memory_snapshot")
            torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
            log.info(f"Memory snapshot saved to {file_prefix}.pickle")
        except Exception as snapshot_err:
            log.error(f"Failed to capture memory snapshot: {snapshot_err}")
        raise e  # Optionally re-raise to exit the script with error


    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    # ReSpinQuant saves per-layer residual rotations (layers.i.R1, layers.i.R2),
    # the head rotation (layers.i.self_attn.R2) and the final basis (R1_final).
    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if key.endswith("R1.weight")
        or key.endswith("R2.weight")
        or key.endswith("R1_final.weight")
    }
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "R.bin")
        torch.save(
            R_dict,
            path,
        )
    dist.barrier()


if __name__ == "__main__":
    train()
