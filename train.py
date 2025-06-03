import os
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import time
import glob
import json

from arc.utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from arc.augmentation import generate_augmentations
from transformers.tokenization_utils_base import PaddingStrategy
from typing import List, Dict, Any

class ARCDataCollator:
    def __init__(self, tokenizer, padding=True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # Pad input_ids
        batch = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels manually and mask pad tokens with -100
        max_len = batch["input_ids"].shape[1]
        padded_labels = torch.full((len(labels), max_len), fill_value=-100)
        for i, label in enumerate(labels):
            length = min(len(label), max_len)
            padded_labels[i, :length] = torch.tensor(label[:length], dtype=torch.long)

        batch["labels"] = padded_labels
        return batch

class ARCDataset(Dataset):
    def __init__(self, raw_examples, tokenizer, pixel_ids, sep_id):
        self.samples = []
        self.tokenizer = tokenizer
        self.pixel_ids = pixel_ids
        self.sep_id = sep_id

        for example in raw_examples:
            try:
                tokens = self.format_prompt(example)
                self.samples.append(tokens)
            except Exception as e:
                print(f"[WARN] Skipping example due to error: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def format_grid(self, grid):
        ids = []
        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep_id)
        return ids

    def format_prompt(self, datapoint):
        sys = self.tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" + system_prompt,
            add_special_tokens=False
        )

        user = self.tokenizer.encode(
            "<|start_header_id|>user<|end_header_id|>\n" + user_message_template1 + "\n",
            add_special_tokens=False
        )

        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)

        user += inp_desc + self.format_grid(datapoint['input'])
        user += out_desc + self.format_grid(datapoint['output'])
        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)
        user += inp_desc + self.format_grid(datapoint['input'])
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)

        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
        messages += assis
        return messages


def load_data_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def train():
    start_time = time.time()

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    token = os.getenv("HF_TOKEN")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    pixel_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)]
    sep_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    print("[INFO] Loading training dataset...")
    train_data = []
    for filepath in glob.glob("/workspace/dataset/*.json"):
        data = load_data_from_json(filepath)
        for example in data:
            train_data.append(example)
            augmented = generate_augmentations([example], example['input'], max_augments=8)
            for aug in augmented:
                train_data.extend(aug['train'])
    print(f"[INFO] Loaded {len(train_data)} raw training examples.")

    dataset = ARCDataset(train_data, tokenizer, pixel_ids, sep_id)
    print(f"[INFO] Loaded {len(dataset)} training examples.")

    training_args = TrainingArguments(
        output_dir="artifacts/checkpoint-final",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        learning_rate=1e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir="artifacts/logs",
        save_strategy="epoch",
        bf16=False,
        fp16=True,
        report_to="none",
    )

    data_collator = ARCDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,  # ✅ Important fix
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model...")
    model.save_pretrained("artifacts/checkpoint-final")
    tokenizer.save_pretrained("artifacts/checkpoint-final")

    duration = (time.time() - start_time) / 60
    print(f"[INFO] Training completed in {duration:.2f} minutes.")


if __name__ == "__main__":
    train()
