# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import argparse
import transformers


@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="test-input", metadata={"help": "Input model"}
    )
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"}
    )
    optimized_rotation_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)"
        },
    )
    respinquant: bool = field(
        default=False,
        metadata={"help": "Train in RespinQuant Manner "}
    )
    lierespinquant: bool = field(
        default=False,
        metadata={
            "help": "Train in LieReSpinQuant manner: instead of learning 2L+1 dense "
                    "D x D bases and compressing the residual transitions with SVD "
                    "afterwards, learn each transition directly as a low-rank "
                    "skew-symmetric Cayley rotation. Exactly orthogonal (and in SO(D)) "
                    "by construction, O(L D r) parameters, and already efficient at "
                    "inference -- nothing is approximated post-hoc. Implies respinquant "
                    "residual-stream layout."
        },
    )
    lie_rank: int = field(
        default=32,
        metadata={"help": "r_max: learned rotation planes per residual transition. "
                          "The skew generator has rank <= 2r."},
    )
    lie_gate_l1: float = field(
        default=0.0,
        metadata={"help": "L1 penalty on the rotation gates gamma. > 0 lets each "
                          "transition settle on its own effective rank instead of a "
                          "uniform r (addresses ReSpinQuant's fixed-rank-per-layer)."},
    )
    lie_learning_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Learning rate for the Cayley factors U/V/gamma. These are ordinary "
                    "Euclidean parameters (orthogonality comes from the parameterization, "
                    "not the optimizer), so they must NOT use the large Stiefel/Cayley-SGD "
                    "step size that --learning_rate carries for R3 (typically ~15)."
        },
    )
    lie_gate_init: float = field(
        default=1e-2,
        metadata={"help": "Std of the initial gates. Near-identity start, so training "
                          "begins from the plain SpinQuant single-rotation solution."},
    )




def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, default=0, help="Random Seed for HuggingFace and PyTorch"
    )

    # Smoothing Arguments
    parser.add_argument(
        "--smooth_quant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Smoothing Technique as implemented in Smoothquant paper",
    )
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path',type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--alpha", type=float, default=0.6,help='migration strength between activation and weight')
    parser.add_argument("--attention",action=argparse.BooleanOptionalAction,default=False,
                        help='Whether to apply smooting technique of attention output vector')
    # Permutation Arguments
    parser.add_argument(
        '--permute',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Apply Permutation Technique')
    parser.add_argument(
        '--permute_mode',
        type=str,
        default='massdiff',choices=['zigzag','random','massdiff']
    )
    parser.add_argument('--permute_seed',type=int,default=-1,help="Random Seed for generationg random permute!!",)
    # Rotation Arguments
    parser.add_argument(
        "--rotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys""",
    )
    parser.add_argument(
        '--offload',
        action=argparse.BooleanOptionalAction,
        default=False,
        help = (
        "When Rotating the model, this option will offload weight tensor to CPU, 2) Apply Matmul in CPU 3) and send rotated weights back to original device"
            )
    )
    # Slider Quant의 아이디어를 바탕으로 하여 실제 R3나 R4를 적용할 Rotation Matrix를 지정 하자
    parser.add_argument(
        "--target_layer_indices", 
        nargs='+',
        type=int,
        default=None,
        help="Hadamard Rotation을 적용할 특정 Decoder 레이어 번호들"
    )

    # Rotation Matrix를 Diagonal하게 적용할 경우
    parser.add_argument(
        '--diagonal',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Apply Block Diagonal Rotation Matrix'
    )
    parser.add_argument(
        '--diagonal_size',
        type=int,
        default = -1,
        help = "Size of block diagonal block is better if this size is a power of 2",
    )
    # parser.add_argument(
    # '--block_size', type=int, default=-1, help='block size of Hadamard matrix in R4 area',
    # )
    # Rotation을 적용할 모드
    parser.add_argument(
        "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
    )
    parser.add_argument(
        "--rotation_seed",
        type=int,
        default=-1,
        help="Random Seed for generating random matrix!!",
    )

    # Selective Rotation 관련 한 Parameter
    parser.add_argument(
        '--deactivate_r1',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deactivate R1 rotation Matrix mentioned in the paper"
    )
    parser.add_argument(
        '--deactivate_r2',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deactivate R2 rotation Matrix mentioned in the SpinQuant paper"
    )
    parser.add_argument(
        '--online_r2', 
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Online R2 as mentioned in QuaRot paper"
    ),
    parser.add_argument(
        '--deactivate_r3',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deactivate R3 rotation Matrix mentioned in the SpinQuant paper"
    ),
    parser.add_argument(
        '--deactivate_r4',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deactivate R4 rotation Matrix mentioned in the SpinQuant paper"
    )
    # ReSpinQuant residual subspace rank (eval only). The `respinquant` flag itself
    # lives on TrainingArguments (single source of truth) and is copied onto ptq_args
    # in process_args_ptq(), so it is intentionally NOT defined here.
    parser.add_argument(
        '--residual_rank',
        type=int,
        default=32,
        help="Rank r of the ReSpinQuant residual subspace approximation (paper default: 32). "
             "Set r >= hidden_size (or r <= 0) to use the EXACT full-rank basis transition "
             "(T_hat == T), which must be lossless at 16-bit."
    )
    parser.add_argument(
        '--deactivate_residual',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Completely skip the ReSpinQuant residual subspace correction (Q/M left as None). "
             "NOTE: with distinct per-layer R1/R2 this leaves a residual-stream basis mismatch, "
             "so PPL is expected to be high even at 16-bit. Use only for A/B comparison."
    )
    parser.add_argument(
        "--fp32_had",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply Hadamard rotation in FP32 (default: False)",
    )
    # Rotation-training checkpointing (optimize_rotation.py). HF Trainer checkpoints
    # dump the FULL model + optimizer state (tens of GB each) into output_dir, which
    # can fill a small disk mid-run and waste the whole training. Default False so a
    # full-disk error cannot happen: the trained rotations are ALWAYS saved separately
    # to output_rotation_path/R.bin regardless of this flag. Pass --save_checkpoints
    # only if you actually need mid-training checkpoints (e.g. to resume long runs).
    parser.add_argument(
        '--save_checkpoints',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save intermediate HF Trainer checkpoints (full model + optimizer state) during "
             "rotation training. Default False to protect against out-of-disk failures; R.bin is "
             "saved separately either way."
    )
    parser.add_argument(
        '--gptq_cpu_offload',
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the CPU-offload / per-layer-streaming GPTQ path (gptq_fwrd_distribute) for "
             "models too large to co-reside with calibration buffers on one GPU (e.g. 70B). "
             "Model is kept on CPU and one decoder layer at a time is streamed to the least-"
             "occupied GPU. Mutually exclusive with --distribute (do NOT use device_map here)."
    )

    parser.add_argument(
        '--per_column',
        action=argparse.BooleanOptionalAction,
        default=False,
        help = (
        "Apply channel-wise quantization. For GEMM, it uses independent scales for "
        "Weight rows [Out_dim] and Activation rows [Token]. This reduces error caused "
        "by outliers in specific channels."
            )
    )

    parser.add_argument(
        '--dynamic_residual_scaling',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FPTQuant's Sn transform (arXiv:2506.04985, Section 3.1.3): move each "
            "RMSNorm so it also normalizes the residual stream, scaling every block's "
            "output by the same per-token factor before the residual add. Function-"
            "preserving without quantization (RMSNorm is scale-invariant, so nothing "
            "upstream changes); with quantization, it reduces outliers at the "
            "down_proj/o_proj input activations, which are otherwise among the worst "
            "quantization bottlenecks. Zero trainable parameters -- safe to combine with "
            "--rotate/--respinquant/--w_rtn/GPTQ with no training step required."
        )
    )

    # Activation Quantization Arguments
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="""Number of bits for inputs of the Linear layers. This will be
                        for all the linear layers in the model (including down-projection and out-projection)""",
    )
    parser.add_argument(
        "--a_groupsize",
        type=int,
        default=-1,
        help="Groupsize for activation quantization. Note that this should be the same as w_groupsize",
    )
    parser.add_argument(
        "--a_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric Activation quantization (default: False)",
    )
    parser.add_argument(
        "--a_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for activation quantization. new_max = max * clip_ratio",
    )

    # Weight Quantization Arguments
    parser.add_argument(
        "--w_bits",
        type=int,
        default=16,
        help="Number of bits for weights of the Linear layers",
    )
    parser.add_argument(
        "--w_groupsize",
        type=int,
        default=-1,
        help="Groupsize for weight quantization. Note that this should be the same as a_groupsize",
    )
    parser.add_argument(
        "--w_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric weight quantization (default: False)",
    )
    parser.add_argument(
        "--w_rtn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Quantize the weights using RtN. If the w_bits < 16 and this flag is not set, we use GPTQ",
    )
    parser.add_argument(
        "--w_auto_clip"
    )
    parser.add_argument(
        "--w_clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""Clipping the weight quantization!
                        We do not support arguments for clipping and we find the best clip ratio during the weight quantization""",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=128,
        help="Number of calibration data samples for GPTQ.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--act_order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="act-order in GPTQ",
    )

    # General Quantization Arguments
    parser.add_argument(
        "--int8_down_proj",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8",
    )

    # KV-Cache Quantization Arguments
    parser.add_argument(
        "--v_bits",
        type=int,
        default=16,
        help="""Number of bits for V-cache quantization.
                        Note that quantizing the V-cache does not need any other rotation""",
    )
    parser.add_argument("--v_groupsize", type=int, default=-1)
    parser.add_argument(
        "--v_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric V-cache quantization",
    )
    parser.add_argument(
        "--v_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for v-cache quantization. new_max = max * clip_ratio",
    )

    parser.add_argument(
        "--k_bits",
        type=int,
        default=16,
        help="""Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries""",
    )
    parser.add_argument("--k_groupsize", type=int, default=-1)
    parser.add_argument(
        "--k_asym",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASymmetric K-cache quantization",
    )
    parser.add_argument(
        "--k_pre_rope",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-RoPE quantization for K-cache (not Supported yet!)",
    )
    parser.add_argument(
        "--k_clip_ratio",
        type=float,
        default=1.0,
        help="Clip ratio for k-cache quantization. new_max = max * clip_ratio",
    )

    # Save/Load Quantized Model Arguments
    parser.add_argument(
        "--load_qmodel_path",
        type=str,
        default=None,
        help="Load the quantized model from the specified path!",
    )
    parser.add_argument(
        "--save_qmodel_path",
        type=str,
        default=None,
        help="Save the quantized model to the specified path!",
    )
    parser.add_argument(
        "--export_to_et",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the quantized model to executorch and save in save_qmodel_path",
    )

    # Experiments Arguments
    parser.add_argument(
        "--capture_layer_io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Capture the input and output of the specified decoder layer and dump into a file",
    )
    parser.add_argument(
        "--layer_idx", type=int, default=10, help="Which decoder layer to capture"
    )
    parser.add_argument(
        "--wikitext2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For Evaluation use wikitext2 evaluation dataset"
    )
    parser.add_argument(
        "--lm_eval_dat", type=str, default=None, choices=["boolq", "arc_easy","arc_challenge","piqa","social_iqa","hellaswag","openbookqa","winogrande"]
    )
    parser.add_argument(
        "--eval_out_path", type=str, default=None, help="path to record result for evaluation"
    )
    parser.add_argument(
        "--stats_path", type=str, default=None, help="path to record stats for Activation"
    )
     # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)

     # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "boolq","social_iqa","openbookqa"],
    )
    parser.add_argument('--lm_eval_batch_size', type=int, default=64, help='Batch size for evaluating with lm eval harness.')
    parser.add_argument(
        "--distribute",
        action="store_true",
        help="Distribute the model on multiple GPUs for evaluation.",
    )

    # Outlier_check Arguments
    parser.add_argument("--weight_check",action=argparse.BooleanOptionalAction,default=False,help="Whether to Store Weight data for profiling")
    parser.add_argument("--act_check",action=argparse.BooleanOptionalAction,default=False,help="Whether to Store activation data for profiling")
    parser.add_argument("--distribution_dir",type=str,default=None,help="Directory for storing weight and activation distribution")
    parser.add_argument("--layer_limit",type=int, default=-1,help="Layer limit of distribution profiling")
    parser.add_argument("--draw",action=argparse.BooleanOptionalAction,default=False,help="Whether to Draw and Save png file")

    # args = parser.parse_args()
    
    args, unknown = parser.parse_known_args()

    # if args.lm_eval:
    #     from lm_eval import tasks
    #     from lm_eval import utils as lm_eval_utils
    #     from lm_eval.tasks import initialize_tasks
    #     initialize_tasks()
    #     for task in args.tasks:
    #         if task not in lm_eval_utils.MultiChoice(tasks.ALL_TASKS):
    #             raise ValueError(f"Invalid task: {task}")
    # assert (
    #     args.a_groupsize == args.w_groupsize
    # ), "a_groupsize should be the same as w_groupsize!"
    assert args.k_pre_rope is False, "Pre-RoPE quantization is not supported yet!"

    return args, unknown


def process_args_ptq():
    ptq_args = None

    ptq_args, unknown_args = parser_gen()

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    if model_args.optimized_rotation_path is not None:
        ptq_args.optimized_rotation_path = model_args.optimized_rotation_path
    else:
        ptq_args.optimized_rotation_path = None
    # Single source of truth: TrainingArguments drives both training and PTQ.
    # LieReSpinQuant uses the ReSpinQuant residual-stream layout, so it implies it.
    ptq_args.lierespinquant = training_args.lierespinquant
    ptq_args.respinquant = training_args.respinquant or training_args.lierespinquant
    ptq_args.lie_rank = training_args.lie_rank
    ptq_args.lie_gate_l1 = training_args.lie_gate_l1
    ptq_args.lie_gate_init = training_args.lie_gate_init
    ptq_args.bsz = training_args.per_device_eval_batch_size

    return model_args, training_args, ptq_args
