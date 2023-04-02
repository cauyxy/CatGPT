import os

import torch
from rm_dataset import create_comparison_dataset, PairwiseDataset
from reward_model import RewardModel
from transformers import AutoTokenizer, Trainer, TrainingArguments
import random


class DataCollatorReward:
    def __call__(self, data):
        batch = {
            "input_ids": torch.cat([f[0] for f in data] + [f[2] for f in data]),
            "attention_mask": torch.cat([f[1] for f in data] + [f[3] for f in data]),
            "labels": torch.tensor([0] * len(data) + [1] * len(data))
        }
        return batch


def compute_metrics(eval_preds):
    chosen_end_scores: torch.Tensor = eval_preds.predictions[0]  # chosen scores
    rejected_end_scores: torch.Tensor = eval_preds.predictions[1]  # rejected scores

    result = {}
    acc = sum(chosen_end_scores > rejected_end_scores) / len(rejected_end_scores)
    result["accuracy"] = acc

    return result


if __name__ == "__main__":
    is_test = True

    sft_path = "../sft/bloomz-1b1-sft"

    tokenizer_path = "bigscience/bloom"
    max_len = 2048

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = "rm_checkpoint"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        logging_steps=10,
        gradient_accumulation_steps=4,
        save_strategy="steps",
        evaluation_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        eval_steps=1000,
        save_steps=1000,
        warmup_steps=100,
        logging_dir="./logs",
        fp16=True,
        bf16=False,
        learning_rate=1e-5,
        deepspeed="ds_config_rm.json",
        save_total_limit=1,
    )

    # Initialize the reward model from the (supervised) fine-tuned GPT-J
    model = RewardModel(sft_path, tokenizer_path)

    # Freeze the first 70% of the hidden layers of the reward model backbone
    layers = model.transformer.h
    num_layers = len(layers)
    num_unfrozen = int(0.3 * num_layers)
    for layer in layers[:-num_unfrozen]:
        layer.requires_grad_(False)

    pairs = create_comparison_dataset()
    pairs = random.sample(pairs, k=30000)
    train_pairs = pairs[:int(len(pairs) * 0.9)]
    val_pairs = pairs[int(len(pairs) * 0.9):]

    train_dataset = PairwiseDataset(train_pairs, tokenizer, max_length=max_len)
    val_dataset = PairwiseDataset(val_pairs, tokenizer, max_length=max_len)

    data_collator = DataCollatorReward()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
