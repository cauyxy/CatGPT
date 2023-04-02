from typing import Callable
import io
import json
from torch.utils.data import Dataset
from tqdm import tqdm


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


# Dahaos/rm-static
class RmStaticDataset(Dataset):
    """
    Dataset for reward model

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        special_token: special token at the end of sentence
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, special_token=None) -> None:
        super().__init__()
        self.chosen = []
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        for data in tqdm(dataset):
            prompt = "Below is an instruction that describes a task. " + \
                     "Write a response that appropriately completes the request.\n\n" + \
                     f"### Instruction:\n{data['prompt']}\n\n### Response:"

            chosen = prompt + data['chosen'] + self.end_token
            chosen_token = tokenizer(chosen,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.chosen.append({
                "input_ids": chosen_token['input_ids'],
                "attention_mask": chosen_token['attention_mask']
            })

            reject = prompt + data['rejected'] + self.end_token
            reject_token = tokenizer(reject,
                                     max_length=max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            self.reject.append({
                "input_ids": reject_token['input_ids'],
                "attention_mask": reject_token['attention_mask']
            })

    def __len__(self):
        length = len(self.chosen)
        return length

    def __getitem__(self, idx):
        return (
            self.chosen[idx]["input_ids"],
            self.chosen[idx]["attention_mask"],
            self.reject[idx]["input_ids"],
            self.reject[idx]["attention_mask"]
        )


def create_comparison_dataset(path="../data/train_pairs.json"):
    dataset = jload(path)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = "Below is an instruction that describes a task. " + \
                 "Write a response that appropriately completes the request.\n\n" + \
                 f"### Instruction:\n{sample['prompt']}\n\n### Response:"
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if len(sample["chosen"]) < 30:
            continue
        pair["chosen"] = prompt + "\n" + chosen_summary
        pair["rejected"] = prompt + "\n" + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )
