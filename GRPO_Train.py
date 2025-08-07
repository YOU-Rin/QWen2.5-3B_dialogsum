from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
import torch
import os
from datasets import load_dataset
from modelscope import snapshot_download
from data_process import dataset_load_and_process
from reward_Func import dialogsum_reward_func

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 4096
lora_rank = 64
model_name = snapshot_download('Qwen/Qwen2.5-3B',  cache_dir='./base_model/Qwen/Qwen2.5-3B')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out"],
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


train_dataset, eval_dataset, test_dataset = dataset_load_and_process()

training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, 
    num_generations = 6, 
    max_prompt_length = 256,
    max_completion_length = 300,
    num_train_epochs=1,
    save_steps = 100,
    max_grad_norm = 0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to = "none",
    output_dir = "outputs/Qwen2.5-3B-GRPO",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
       dialogsum_reward_func
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)
trainer.train()