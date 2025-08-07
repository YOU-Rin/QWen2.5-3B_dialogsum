from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, load_from_disk
import torch
import os
from data_process import dataset_load_and_process_PEFT, tokenize_data
from modelscope import snapshot_download

#设置镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_name = snapshot_download('Qwen/Qwen2.5-3B')
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载模对应的分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

save_dir="data_cache"
# 加载数据集
tokenizer_dataset = dataset_load_and_process_PEFT(tokenizer)
#tokenizer_dataset.save_to_disk(os.path.join(save_dir, "Tokenizerd_Dataset"))

train_dataset = tokenizer_dataset['train']
eval_dataset = tokenizer_dataset['validation']
test_dataset = tokenizer_dataset['test']
print(train_dataset)  # 打印第一个样本以检查数据格式



#加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 这是进行4比特量化，即使用QLoRA
    device_map="auto",
    trust_remote_code=True,
)

model = prepare_model_for_kbit_training(model)

# 配置 LoRA 参数
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# 配置训练参数
training_args = TrainingArguments(
    output_dir="outputs/Qwen2.5-3B-PEFT",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    do_eval=True,
    eval_steps=500,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
    report_to='none',
    remove_unused_columns=False,
    
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)
trainer.train()




