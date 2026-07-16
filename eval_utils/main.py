# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers


from train_utils import apply_r3_r4
from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils,smooth_quant,permute
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)


def ptq_model(args, model, log, tokenizer, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()

    # Smoothing Applied if requested
    if args.smooth_quant:
        args.act_scales = f"./act_scales/{model_args.input_model.split('/')[-1]}.pt"
        log.info("Smoothing Applied")
        # print(f"smoothing Alpha: {args.alpha}")
        # if args.attention:
        #     print("Smoothing Applied to Attention")
        act_scales = torch.load(args.act_scales)
        smooth_quant.smoothing(model,args,act_scales)
    if args.permute:
        permute.permute(model,args,log,tokenizer)
    # Rotate the weights
    # log.info("LayerNorm Fusion Applied For R1 Transform")
    # fuse_norm_utils.fuse_layer_norms(model)
    if args.rotate:
        log.info("R1: {}, R2: {}, R3: {}, R4: {}".format(
            not args.deactivate_r1, 
            not args.deactivate_r2, 
            (not args.deactivate_r3 and args.k_bits < 16), # R3: KV 캐시 양자화 시에만 의미 있음
            not args.deactivate_r4                         # R4: 쉼표로 구분 필수
        ))

        
        if not args.deactivate_r1:
            fuse_norm_utils.fuse_layer_norms(model) #
            log.info("LayerNorm Fusion Applied For R1 Transform")

        
        rotation_utils.rotate_model(model, args,model_args)
        if not args.deactivate_r4:
            log.info("Applying R4 Transform")
            apply_r3_r4.rotate_model(model, args) # 실제로 R4 Rotation만 적용함

        if args.online_r2:
            log.info("Online R2 rotation in QuaRot Applied")

        utils.cleanup_memory(verbos=True)

        quant_utils.add_actquant(model,percolumn=args.per_column)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model) # Quantized 된 layer의 dictionary format을 만든 {name: ActQuantWrapper}
        # print(qlayers)
        for name in qlayers:
            # print(name)
            # 1. 현재 레이어의 인덱스 추출 (예: 'model.layers.5.mlp.down_proj' -> 5)
            try:
                current_idx = int(name.split('.')[2])
            except (IndexError, ValueError):
                continue
            
            # 2. 지정된 인덱스 리스트에 포함되는지 확인
            is_target = True
            if args.target_layer_indices is not None:
                if current_idx not in args.target_layer_indices:
                    is_target = False

            # if not is_target:
            #     continue
            if is_target:    
                if "down_proj" in name and not args.deactivate_r4:
                    if not args.diagonal and not args.permute:
                        try: 
                            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size) # output1: 2의 제곱이 아닌 hadamard Matrix: walsh hadamard matrix가 아닌 Sloane Hadamard Matrix인 경우, K: Sloane Hadamard Matrix의 Size
                            qlayers[name].online_full_had = True
                            qlayers[name].had_K = had_K
                            qlayers[name].K = K
                            qlayers[name].fp32_had = args.fp32_had
                        except:
                            had_dim = hadamard_utils.largest_power_of_two_divisor(model.config.intermediate_size)

                            had_K, K = hadamard_utils.get_hadK(had_dim)

                            qlayers[name].online_diagonal_had = True
                            qlayers[name].had_K = had_K
                            qlayers[name].K = K
                            qlayers[name].had_dim = had_dim
                            qlayers[name].fp32_had = args.fp32_had

                            log.warning(
                                f"{name}: full Hadamard is not supported for "
                                f"intermediate_dim={model.config.intermediate_size}. "
                                f"Falling back to block Hadamard with "
                                f"block_size={had_dim}, "
                                f"num_blocks={model.config.intermediate_size // had_dim}."
                            )
                        # qlayers[name].transpose=True
                    else:
                        had_K, K = hadamard_utils.get_hadK(args.diagonal_size) # output1: 2의 제곱이 아닌 hadamard Matrix, # Output2: 해당 matrix의 차원수 (shape?) 
                        qlayers[name].online_diagonal_had = True
                        qlayers[name].had_K = had_K
                        qlayers[name].K = K
                        qlayers[name].had_dim = args.diagonal_size
                        qlayers[name].fp32_had = args.fp32_had
                
            if 'o_proj' in name and args.online_r2:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
    else:
        log.info("Rotation Not applied")
        quant_utils.add_actquant(
            model, percolumn=args.per_column
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                # layer_groupsize = head_dim
                layer_groupsize = model_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            log.info("Applying R3")
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            if args.rotate and not args.offline and not args.deactivate_r3:
                for layer in layers:
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward( # 해당 instance만 apply_rotary_pos_emb가 QkrotationWrapper의 binding되도록 할당함
                        layer.self_attn,
                        rope_function_name,
                        config=model.config, 
                        **k_quant_config,
                    )

    return model
