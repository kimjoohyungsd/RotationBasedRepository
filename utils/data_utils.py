# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import random
from typing import Any, Dict

import datasets
import torch
import transformers


def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if eval_mode:
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt") # 모든 test의 data들을 \n\n을 추가한후 tokenize한다 
        return testenc
    else:
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j] # Input text 추출
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader



class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d)) # 일련의 Dataset을 tokenize한다 output type

        grouped_dataset = self.group_texts(tokenized_datasets) # group_texts 
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples): # examples: tokenized_datasets:
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = { # input_ids: []
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


# Outlier check를 하기 위한 Wikitext dataset을 진행한다
def load_example_dataset(dataset_name="wikitext", sample_size=128):
        """Load example dataset for activation analysis"""
        try:
            # Different options for datasets
            if dataset_name == "wikitext":
                print(f"Loading wikitext dataset (sample size: {sample_size})...")
                dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                texts = dataset["text"][:sample_size]
                # Filter out empty texts
                texts = [text for text in texts if len(text.strip()) > 50]
                
            elif dataset_name == "code":
                print(f"Loading code dataset (sample size: {sample_size})...")
                dataset = datasets.load_dataset("codeparrot/github-code", split="train")
                texts = [sample["code"] for sample in dataset[:sample_size]]
                
            elif dataset_name == "multilingual":
                print(f"Loading multilingual dataset (sample size: {sample_size})...")
                # Custom multilingual text including Korean
                texts = [
                    "자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있습니다.",
                    "自然言語処理モデルの重み分布を分析しています。このテキストには様々な言語パターンが含まれています。",
                    "Analyzing the weight distribution of natural language processing models. This text contains various language patterns.",
                    "Estamos analizando la distribución de pesos en modelos de procesamiento de lenguaje natural.",
                    "Мы анализируем распределение весов в моделях обработки естественного языка."
                ][:sample_size]
                
            else:  # Default to custom text
                print("Using default sample text...")
                texts = [
                    """자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있어서 
                    모델이 처리하는 방식을 관찰하기 좋습니다. The distribution of weights in language models can show 
                    interesting patterns across different components like attention and feed-forward networks."""
                ]
                
            print(f"Loaded {len(texts)} text samples")
            return texts
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using default sample text instead")
            return [
                """자연어 처리 모델의 가중치 분포를 분석하는 중입니다. 이 텍스트는 다양한 언어 패턴을 포함하고 있어서 
                모델이 처리하는 방식을 관찰하기 좋습니다. The distribution of weights in language models can show 
                interesting patterns across different components like attention and feed-forward networks."""
            ]
