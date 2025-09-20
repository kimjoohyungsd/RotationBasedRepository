import torch
import torch.distributed as dist
import transformers
from transformers import AutoModel,LlamaForCausalLM,LlamaTokenizer,LlamaTokenizerFast
from utils import fuse_norm_utils,hadamard_utils
from train_utils import apply_r3_r4,rtn_utils

import datetime
import logging
from logging import Logger

from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils import rotation_utils,gptq_utils
from utils import hadamard_utils,fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils import data_utils, eval_utils, utils
from utils.utils import HadamardTransform
from utils.process_args import process_args_ptq

import transformers

from scipy.linalg import hadamard
import numpy as np

import os
import subprocess

import math
import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple



class ActWrapper (torch.nn.Module): # Activation의 Dynamic Rotation을 적용해 주는 함수
    def __init__(self,module: torch.nn.Linear) -> None:
        super(ActWrapper,self).__init__()
        self.module = module
        self.weight = module.weight
        self.bias   = module.bias
        self.register_buffer("had_K",torch.tensor(0))
        self._buffers["had_K"] = None
        self.K=1
        self.online_full_had=False
        self.online_partial_had=False
        self.had_dim=0
        self.fp32_had= True

    def forward(self, x, R1=None, R2=None, transpose=False):
        x_dtype = x.dtype
    
        # Rotate, if needed
        if self.online_full_had:
            x=x.cuda()
            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K.cuda(), self.K,transpose).to(
                    x_dtype
                )
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K.cuda(), self.K,transpose)
        x=x.cuda()
        x = self.module(x)

        return x

# Corrected wrapping code in main function
def apply_activation_wrapper(model, had_K, K):
    """Apply activation wrapper to all down_proj layers"""
    
    layers = list(model.model.layers)
    for i, layer in enumerate(layers):
        print(f"Wrapping layer {i} down_proj with ActWrapper")
        
        # Step3: Add Activation Wrapper
        original_module = layer.mlp.down_proj
        
        # Create wrapper
        wrapper = ActWrapper(original_module)
        
        # Set hadamard parameters
        wrapper.had_K = had_K
        wrapper.K = K  # Fixed: use uppercase K to match
        wrapper.online_full_had = True
        wrapper.fp32_had = True
        
        # Replace the original module with wrapper
        layer.mlp.down_proj = wrapper  # Fixed: correct attribute name
        
        print(f"  Layer {i} - Wrapper applied successfully")
        print(f"  Layer {i} - had_K shape: {wrapper.had_K.shape}")
        print(f"  Layer {i} - K value: {wrapper.K}")


# def parser_gen():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--optimized_rotation_path",
#         type=str,
#         default=None,
#         help="location of Rotation Matrix path"
#     )
#     parser.add_argument(
#         "--output_path",
#         type=str,
#         default=None,
#         help = "location of output model"
#     )
#     parser.add_argument(
#         "--rotate_mode", type=str, default="hadamard", choices=["hadamard", "random"]
#     )

#     parser.add_argument(
#         "--r4",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help="Apply r4 rotational Matrix"
#     )

#     parser.add_argument(
#         '--test', action=argparse.BooleanOptionalAction,
#         default=False,
#         help= 'Print with test example'
#     )
#     parser.add_argument(
#         '--quanitze', action=argparse.BooleanOptionalAction,
#         default = False,
#         help= "Apply Fake Quantization to model"
#     )
#     parser.add_argument(
#         '--w_rtn', action=argparse.BooleanOptionalAction,
#         default = False,
#         help = "When quantizing apply round to nearest option"
#     )
#     args = parser.parse_args()

#     return args

log: Logger = utils.get_logger("spinquant")
def main() -> None:

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8)) # initializes the default distributed process group and Communication backend: NCCL
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank() # 

    log.info("the rank is {}".format(local_rank)) # 두번 log
    torch.distributed.barrier() # OS에서 Barrier 설정과 동일 

    config = transformers.AutoConfig.from_pretrained( 
        model_args.input_model, token=model_args.access_token
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=model_args.input_model,config=config,torch_dtype=dtype)
    model.seqlen = training_args.model_max_length
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(model_args.input_model)

    if(ptq_args.rotate):
        # Step1: Fuse layer norms
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model,ptq_args)

         # Step3: Apply R4 Inverse matrix to each down projection layer
        # if (not ptq_args.no_r4):
        #     apply_r3_r4.rotate_model(model,args)


    # RTN quantization을 적용한다 
    if(ptq_args.w_rtn):
        gptq_utils.rtn_fwrd(model,"cuda",ptq_args)
        
    # Step4: Testing
        
    # if (not ptq_args.no_r4):
    #     had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    #     apply_activation_wrapper(model, had_K, K)
    #     print("R4 matrix fusion and activation wrapper applied successfully!")

    # test_input = "Hello, how are you? Would you like to play baseball with me?"
    # inputs = tokenizer(test_input, return_tensors="pt")
    
    # # Make sure input is on the right device
    # # For multi-GPU models, typically the first layer's device is used for inputs
    # input_device = model.model.embed_tokens.weight.device
    # inputs = {k: v.to(input_device) for k, v in inputs.items()}
    
    # print(f"Input device: {input_device}")
    
    # Generate with the model
    # with torch.no_grad():
    #     # generated_ids = model.generate(
    #     #     inputs['input_ids'],
    #     #     max_length=50,
    #     #     do_sample=False,
    #     #     pad_token_id=tokenizer.eos_token_id
    #     # )
    
    #     # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     # print(f"Generated text: {generated_text}")
    #     # tokenizer = LlamaTokenizerFast.from_pretrained(
    #     # pretrained_model_name_or_path=args.input_model,
    #     # cache_dir=training_args.cache_dir,
    #     # model_max_length=training_args.model_max_length,
    #     # padding_side="right",
    #     # use_fast=True,
    #     # add_eos_token=False,
    #     # add_bos_token=False,
    #     # token=model_args.access_token,
    #     #  )
    #     model.cuda()
    #     # log.info("Complete tokenizer loading...")
    #     model.config.use_cache = False

    #     testloader = data_utils.get_wikitext2(
    #         seed=ptq_args.seed,
    #         seqlen=2048,
    #         tokenizer=tokenizer,
    #         eval_mode=True,
    #     )

    #     dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    #     log.info("wiki2 ppl is: {}".format(dataset_ppl))
    #     dist.barrier()
    #     # print("wiki2 ppl is {}".format(dataset_ppl))
    #     # print("Model verification completed successfully!")

    # Step3: Save to output path
    if ptq_args.save_qmodel_path is not None:
        print(f"Saving to {ptq_args.save_qmodel_path}!")
        torch.save(model.state_dict(),ptq_args.save_qmodel_path)

    # Step4: Testing 
    model.seqlen = training_args.model_max_length
    
    testloader = data_utils.get_wikitext2(
            seed=ptq_args.seed,
            seqlen=2048,
            tokenizer=tokenizer,
            eval_mode=True,
        )

    # Step5 : Apply R4 rotation in matrix
    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
    apply_activation_wrapper(model, had_K, K)
    print("R4 matrix fusion and activation wrapper applied successfully!")

    model=model.cuda()
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, ptq_args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()
        # print("wiki2 ppl is {}".format(dataset_ppl))
        # print("Model verification completed successfully!")
if __name__ == "__main__":
    main()


