from typing import List

import torch
from reward_model.reward_model import RewardModel
from transformers import AutoTokenizer
from dataset.sft_dataset import get_dataset_from_jsonl

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig


TOKENIZER_PATH = "bigscience/bloom"
SFT_MODEL_PATH = "./sft/bloomz-1b1-sft"
REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"

UNFROZEN_LAYER = 10
MAX_NEW_TOKENS = 500

config = TRLConfig(
    train=TrainConfig(
        seq_length=2048,
        epochs=50,
        total_steps=10000,
        batch_size=4,
        checkpoint_interval=1000,
        eval_interval=200,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
    ),
    model=ModelConfig(
        model_path=SFT_MODEL_PATH,
        num_layers_unfrozen=UNFROZEN_LAYER,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=TOKENIZER_PATH,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 5.0e-6,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.01,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 100000,
            "eta_min": 5.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=128,
        chunk_size=16,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.2,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": MAX_NEW_TOKENS,
        },
    ),
)


def train():
    # Load the pre-trained reward model
    rw_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    rw_tokenizer.pad_token = rw_tokenizer.eos_token
    rw_model = RewardModel(SFT_MODEL_PATH, TOKENIZER_PATH)
    rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
    rw_model.half()
    rw_model.eval()
    rw_device = torch.device("cuda:7")  # set reward model device
    rw_model.to(rw_device)

    def get_scores(samples: List[str]):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i: i + batch_size]
            sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = rw_tokenizer(
                sub_samples,
                truncation=True,
                max_length=config.train.seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(rw_device)
            attn_masks = encodings_dict["attention_mask"].to(rw_device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def reward_fn(samples: List[str], **kwargs):
        scores = get_scores(samples)
        return scores

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    data_path = "../data/only_prompts.json"
    eval_ratio = 0.1
    prompts, targets = get_dataset_from_jsonl(data_path, tokenizer=tokenizer)

    trainer = trlx.train(
        reward_fn=reward_fn,
        prompts=prompts[int(len(prompts)*eval_ratio):],
        eval_prompts=prompts[:int(len(prompts)*eval_ratio)],
        config=config,
    )
    trainer.save_pretrained("./ppo")


if __name__ == '__main__':
    train()
