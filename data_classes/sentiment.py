import json
from torch.utils.data import Dataset
import random
import numpy as np
from utils import EditBatchSampler, dict_to, build_distr_matrix
import torch

class SentimentDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path,
        config,
        max_length=96,
    ):
        super().__init__()
        self.tok = tokenizer
        self.config = config
        self.max_length = max_length

        with open(data_path) as f:
            self.data = json.load(f)

        self.templates = [
            "What do you think of {}?",
            "What do you feel about {}?",
            "How do you view {}?",
        ]
        for position in [
            "opinion of",
            "stance on",
            "position on",
            "attitude about",
            "view on",
            "take on",
            "impression of",
            "assessment of",
            "judgment of",
            "sentiment of",
        ]:
            self.templates.append("What is your " + position + " {}?")

        self.loc_distr_matrix, self.loc_idx_matrix = None, None
        if self.config.data.hard_neg and "train" in data_path:
            edit_qs = [sample["ent"] for sample in self.data]
            self.loc_distr_matrix, self.loc_idx_matrix = build_distr_matrix(edit_qs, config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def sample_completions(self, idx, do_sample=True):
        sample = self[idx]
        inner_sent = random.choice([-1, 1])
        # inner_comp = [random.choice(sample["pos"]) if inner_sent == 1 else random.choice(sample["neg"])]
        inner_comp = ["Sentiment: " + ("Positive" if inner_sent == 1 else "Negative")]
        # inner_temp = random.choice(self.templates)
        # inner_prompt = [inner_temp.format(sample["ent"])]
        inner_prompt = ["Topic: " + sample["ent"]]

        if do_sample:
            # Sample 1 completion
            outer_sent = [inner_sent]
            outer_comp = [random.choice(sample["pos"]) if inner_sent == 1 else random.choice(sample["neg"])]
            outer_temp = random.choice(self.templates)
            outer_prompt = [outer_temp.format(sample["ent"])]
        else:
            # Use all possible completions for outer
            outer_sent = ([-1] * len(sample["neg"])) if inner_sent == -1 else ([1] * len(sample["pos"]))
            outer_comp = sample["neg"] if inner_sent == -1 else sample["pos"]
            outer_temp = random.choices(self.templates, k=len(outer_sent))
            outer_prompt = [t.format(sample["ent"]) for t in outer_temp]

        all_sent = ([-1] * len(sample["neg"])) + ([1] * len(sample["pos"]))
        all_comp = sample["neg"] + sample["pos"]
        all_temp = random.choices(self.templates, k=len(all_sent))
        all_prompt = [t.format(sample["ent"]) for t in all_temp]

        return {
            "ent": sample["ent"],
            "inner_prompt": inner_prompt,
            "inner_comp": inner_comp,
            "inner_sent": inner_sent,
            "outer_prompt": outer_prompt,
            "outer_comp": outer_comp,
            "all_prompt": all_prompt,
            "all_sent": all_sent,
            "all_comp": all_comp,
        }

    def get_edit_labels(self, ids, prompts=None):
        labels = ids.clone()
        labels[labels == self.tok.pad_token_id] = -100
        return labels

    def collate_fn(self, batch):
        inner_prompt = [prompt for b in batch for prompt in b["inner_prompt"]]
        inner_comp = [comp for b in batch for comp in b["inner_comp"]]
        outer_prompt = [prompt for b in batch for prompt in b["outer_prompt"]]
        outer_comp = [comp for b in batch for comp in b["outer_comp"]]
        all_prompt = [prompt for b in batch for prompt in b["all_prompt"]]
        all_comp = [comp for b in batch for comp in b["all_comp"]]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "inner_q": inner_prompt,
                "inner_a": inner_comp,
                "outer_q": outer_prompt,
                "outer_a": outer_comp,
                "all_q": all_prompt,
                "all_a": all_comp,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        batches["all_sent"] = [s for b in batch for s in b["all_sent"]]
        batches["inner_sent"] = [b["inner_sent"] for b in batch for s in b["all_sent"]]
        batches["raw"] = batch

        pos_pairs = []
        for idx, b in enumerate(batch):
            for _ in range(len(b["all_prompt"])):
                pos_pairs.append([len(pos_pairs), idx])

        batches["pos_pairs"] = torch.LongTensor(pos_pairs)
        return batches

    def edit_generator(self, batch_size, n=None, do_sample=False):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(
            n,
            memorize_mode=self.config.single_batch,
            loc_disjoint=True,
            seed=self.config.seed,
            hard_neg=self.config.data.hard_neg,
            hard_neg_prob=self.config.data.hard_neg_prob,
            loc_distr_matrix=self.loc_distr_matrix,
            loc_idx_matrix=self.loc_idx_matrix,
        )

        while True:
            edit_idxs, loc_idxs = sampler.sample(batch_size)
            edit_toks = self.collate_fn([self.sample_completions(idx, do_sample) for idx in edit_idxs])
            loc_toks = self.collate_fn([self.sample_completions(idx, do_sample) for idx in loc_idxs])

            edit_inner = {
                "input_ids": edit_toks["inner_q_input_ids"],
                "attention_mask": edit_toks["inner_q_attention_mask"],
                "labels": self.get_edit_labels(edit_toks["inner_a_input_ids"]),
                "decoder_input_ids": edit_toks["inner_a_input_ids"],
                "decoder_attention_mask": edit_toks["inner_a_attention_mask"],
            }

            edit_outer = {
                "input_ids": edit_toks["all_q_input_ids"],
                "attention_mask": edit_toks["all_q_attention_mask"],
                "labels": self.get_edit_labels(edit_toks["all_a_input_ids"]),
                "decoder_input_ids": edit_toks["all_a_input_ids"],
                "decoder_attention_mask": edit_toks["all_a_attention_mask"],
            }

            loc = {
                "input_ids": loc_toks["inner_q_input_ids"],
                "attention_mask": loc_toks["inner_q_input_ids"],
                "labels": self.get_edit_labels(loc_toks["inner_a_input_ids"]),
                "decoder_input_ids": loc_toks["inner_a_input_ids"],
                "decoder_attention_mask": loc_toks["inner_a_attention_mask"],
            }

            pos_pairs = edit_toks["pos_pairs"]

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "outer_sent": edit_toks["all_sent"],
                "inner_sent": edit_toks["inner_sent"],
                "loc": loc,
                "cond": edit_inner,
                "pos_pairs": pos_pairs,
            }

            yield dict_to(batch, self.config.device)


def default_dataset():
    import transformers
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.device = "cpu"
    config.single_batch = False
    config.data = SimpleNamespace()
    config.data.n_edits = 10
    config.data.hard_neg = True
    config.data.hard_neg_prob = 0.5
    config.data.hard_neg_neighbors = 10
    config.data.hard_neg_exclude = 1
    config.data.hard_neg_temp = 0.1
    config.single_batch = False
    config.seed = 0
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/blenderbot-90M')
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer, SentimentDataset(tokenizer, "data/sentiment/blender_train.json", config)


if __name__ == "__main__":
    tok, ds = default_dataset()
    edit_gen = ds.edit_generator(20)
    batch = next(edit_gen)
    breakpoint()
