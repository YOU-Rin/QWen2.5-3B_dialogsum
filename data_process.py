from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# 设置模型名称和设备
model_name = "./Qwen/Qwen2.5-0.5B-Instruct-local"
device = "cuda" if torch.cuda.is_available() else "cpu"

## GRPO
def dataset_load_and_process():
    # 定义训练集和测试集
    dataset = load_dataset("./dialogsum")

    # 定义转换函数：将对话和摘要拼接成prompt和response
    def convert_to_prompt_response(example):
        prompt = f"Summarize the following dialog:\n{example['dialogue']}"
        response = example["summary"]
        return {"prompt": prompt, "response": response}

    # 对数据集 map 映射转换
    dataset = dataset.map(convert_to_prompt_response)


    # 对数据集 map 映射转换
    formatted_dataset = dataset.map(convert_to_prompt_response)

    # 数据集中有三种不同的划分：train、validation 和 test
    train_dataset = formatted_dataset["train"]
    eval_dataset = formatted_dataset["validation"]
    test_dataset = formatted_dataset["test"]
    
    return train_dataset, eval_dataset, test_dataset

from transformers import AutoTokenizer
from datasets import load_dataset

## PEFT , 这个需要手动进行tokenizer处理后才能用于模型训练
def dataset_load_and_process_PEFT(tokenizer, max_length=1024):
    # 加载原始数据集
    dataset = load_dataset("./dialogsum")

    # 转换函数：拼接 prompt 和 response
    def convert_to_prompt_response(example):
        prompt = f"Summarize the following dialog:\n{example['dialogue']}"
        response = example["summary"]
        return {"prompt": prompt, "response": response}

    dataset = dataset.map(convert_to_prompt_response)

    # tokenizer 处理函数
    def tokenize_function(example):
        # 把 prompt 和 response 合并，作为输入
        full_prompt = example["prompt"] + "\n" + example["response"]

        tokenized = tokenizer(
            full_prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        # 把 labels 设置为 input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset["train"].column_names)

    return tokenized_dataset

# PEFT
def tokenize_data(tokenizer):
    
    dataset = load_dataset("./dialogsum")

    def tokenize_function(example):
        start_prompt = 'Summarize the following conversation.\n\n'
        end_prompt = '\n\nSummary: '
        prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
        example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
        example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
        return example

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets['train'] = tokenized_datasets['train'].remove_columns(['id', 'topic', 'dialogue', 'summary',])
    tokenized_datasets['validation'] = tokenized_datasets['validation'].remove_columns(['id', 'topic', 'dialogue', 'summary',])

    return tokenized_datasets

# def tokeninze_function(example):
#     start_prompt = 'Summarize the following conversation. \n\n'
#     end_prompt = '\n\nSummary: '
#     prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
#     example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, 
#                                      return_tensors='pt').input_ids
#     example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True, 
#                                  return_tensors='pt').input_ids
    
#     return example

# The Dataseta ctually contains 3 diff splits: train, validation, and test.
# The tokenize_function code is handling all data across all splits in batches
# tokenize_datasets = dataset.map(tokeninze_function, batched=True)
# tokenize_datasets = tokenize_datasets.remove_columns(['id', 'topic', 'dialogue',
#                                                      'summary'])

# 查看转换后的示例
"""
print(dataset)
DatasetDict({
    train: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic', 'prompt', 'response'],
        num_rows: 12460
    })
    validation: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic', 'prompt', 'response'],
        num_rows: 500
    })
    test: Dataset({
        features: ['id', 'dialogue', 'summary', 'topic', 'prompt', 'response'],
        num_rows: 1500
    })
}).
"""





