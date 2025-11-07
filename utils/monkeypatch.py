# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import functools
import types


def copy_func_with_new_globals(f, globals=None):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    if globals is None:
        globals = f.__globals__
    g = types.FunctionType(
        f.__code__,
        globals,
        name=f.__name__,
        argdefs=f.__defaults__,
        closure=f.__closure__,
    )
    g = functools.update_wrapper(g, f)
    g.__module__ = f.__module__
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g


def add_wrapper_after_function_call_in_method(
    module, # layer.self_attn
    method_name, #"forward"
    function_name, # "apply_rotary_emb"
    wrapper_fn, #
):
    """
    This function adds a wrapper after the output of a function call in the method named `method_name`.
    Only calls directly in the method are affected. Calls by other functions called in the method are not affected.
    """

    original_method = getattr(module, method_name).__func__ ## Step1: original forward method값을 가지고 온다
    method_globals = dict(original_method.__globals__) # forwardmethod가 가지고 올 수 있는 모든 namespace의 table들을 가지고 온다
    wrapper = wrapper_fn(method_globals[function_name]) # QKrotationWrapper module class 안에서 function_name을 argument로 받아 Initialize 한다  
    method_globals[function_name] = wrapper # forward pass method의 globalnamespace에 Apply_rotary_positonal에 해당하는 table을 QKrotationWrapper의  Instance 로 바꾼다
    new_method = copy_func_with_new_globals(original_method, globals=method_globals) # 변화된 forward pass 함수를 생성
    setattr(module, method_name, new_method.__get__(module)) # 일반 생성된 python 함수로 self_attn.forward 함수가 이 함수에 binding됨
    return wrapper
