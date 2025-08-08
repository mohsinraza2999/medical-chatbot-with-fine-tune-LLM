import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, logging
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd

def load_data(path: str = "data/data.csv", nrows=1000) -> Dataset:
    df = pd.read_csv(path, nrows=nrows)
    df.columns = [str(q).strip() for q in df.columns]
    return Dataset.from_pandas(df)

def load_model_and_tokenizer(model_name: str, bnb_config, device_map):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

def train():
    model_name = "NousResearch/Llama-2-7b-chat-hf"
    new_model = "LLama-2-medical-chatbot"
    device_map = {"": 0}
    output_dir = "./results"

    data = load_data()

    bnb_config = get_bnb_config()
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config, device_map)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

if __name__ == "__main__":
    logging.set_verbosity(logging.CRITICAL)
    train()