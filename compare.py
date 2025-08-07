from transformers import AutoTokenizer, GenerationConfig, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import torch
import os
import evaluate  ## for calculating rouge score
from data_process import dataset_load_and_process
from modelscope import snapshot_download

# 加载原始模型
model_name = snapshot_download('Qwen/Qwen2.5-3B',)
peft_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 给原始模型中注入微调权重
peft_model = PeftModel.from_pretrained(peft_model,
                                      'outputs/Qwen2.5-3B-GRPO/checkpoint-2077',
                                      torch_dtype=torch.bfloat16,
                                      is_trainable=False) ## is_trainable mean just a forward pass jsut to get a sumamry

original_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)



train_dataset, eval_dataset, test_dataset = dataset_load_and_process()

# 选择一个对话进行验证
index = 2
dialogue = eval_dataset[index]['dialogue'] # 从diindex个样本中提取信息
human_baseline_summary = eval_dataset[index]['summary']



# 设置提示词模板
prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors='pt').input_ids

original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)


peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)


# 输出摘要结果
print(f'Human Baseline summary: \n{human_baseline_summary}\n')
print(f'Original Model Output \n{original_model_text_output}\n')
print(f'Peft Model Output \n{peft_model_text_output}\n')

# 计算 rouge 分数
rouge = evaluate.load('rouge')
original_model_results = rouge.compute(predictions=[original_model_text_output], 
                                       references=[human_baseline_summary],
                                      use_aggregator=True,
                                      use_stemmer=True)

peft_model_results = rouge.compute(predictions=[peft_model_text_output], 
                                    references=[human_baseline_summary],
                                    use_aggregator=True,
                                    use_stemmer=True)

print(f'Original Model: \n{original_model_results}\n') 
print(f'PEFT Model: \n{peft_model_results}\n') 
