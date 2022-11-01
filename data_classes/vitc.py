import torch
from torch.utils.data import Dataset
from utils import EditBatchSampler, dict_to
import random
import json
import jsonlines
import logging
from collections import Counter

LOG = logging.getLogger(__name__)


ENTAILED = "SUPPORTS"
CONTRADICTED = "REFUTES"
NEUTRAL = "NOT ENOUGH INFO"


def map_labels(str_labels):
    fc_label_map = {
        ENTAILED: 1,
        CONTRADICTED: 0,
        NEUTRAL: -100
    }

    return [fc_label_map[label] for label in str_labels]


class VitC(Dataset):
    def __init__(self, data_path, split, tokenizer, config, mutex=True):
        if split.startswith("val"):
            split = "dev"

        self.split = split
        self.tok = tokenizer
        self.config = config

        path = f"{data_path}/{split}.jsonl"
        self.data = []
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                if obj["revision_type"] == "real" and len(obj["evidence"].split(' ')) > 10:
                    obj["inner_label"] = ENTAILED
                    self.data.append(obj)
        random.shuffle(self.data)
        if mutex:
            self.mutex = [ascii(obj["page"]) for obj in self.data]
        else:
            self.mutex = None

        labels = [d["label"] for d in self.data]
        counter = Counter(labels)
        LOG.warning(counter)

        self.keep_probs = None
        if split != "test":
            counts = [counter[ENTAILED], counter[CONTRADICTED], counter[NEUTRAL]]
            keep_probs = [float(min(counts)) / c for c in counts]
            label_keep_map = {ENTAILED: keep_probs[0], CONTRADICTED: keep_probs[1], NEUTRAL: keep_probs[2]}
            self.keep_probs = [label_keep_map[d["label"]] for d in self.data]
            LOG.info(labels[:5])
            LOG.info(self.keep_probs[:5])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        def sample_loc(b):
            loc_obj = random.choice(self.data)
            while loc_obj["page"] in b["page"] or b["page"] in loc_obj["page"]:
                loc_obj = random.choice(self.data)
            return loc_obj

        inner = [b["evidence"] for b in batch]
        pos_pairs = []
        for idx, b in enumerate(batch):
            if b["label"] != NEUTRAL:
                pos_pairs.append([len(pos_pairs), idx])
        outer = [b["claim"] for b in batch]
        loc_objs = [b if b["label"] == NEUTRAL else sample_loc(b) for b in batch]
        loc = [b["claim"] for b in loc_objs]
        loc_pages = [b["page"] for b in loc_objs]

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "inner": inner,
                "outer": outer,
                "loc": loc,
                "cond": inner,
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=120,
                truncation=True,
            ).items()
        }

        for k, v in batches.items():
            if k.startswith("outer_"):
                idxs = [idx for idx in range(len(batch)) if batch[idx]["label"] != NEUTRAL]
                batches[k] = v[idxs]

        outer_labels = torch.LongTensor(map_labels([b["label"] for b in batch if b["label"] != NEUTRAL])).unsqueeze(-1)
        batches["outer_labels"] = outer_labels
        batches["pos_pairs"] = torch.LongTensor(pos_pairs)
        inner_labels = torch.LongTensor(map_labels([b["inner_label"] for b in batch])).unsqueeze(-1)
        batches["inner_labels"] = inner_labels
        batches["pages"] = [b["page"] for b in batch]
        batches["loc_pages"] = loc_pages
        batches["hard_pos_mask"] = [True] * outer_labels.shape[0]
        batches["hard_neg_mask"] = [b["label"] == NEUTRAL for b in batch]
        return batches

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(
            n,
            memorize_mode=self.config.single_batch,
            loc_disjoint=True,
            seed=self.config.seed,
            keep_probs=self.keep_probs,
            mutex=self.mutex,
        )

        while True:
            idxs, _ = sampler.sample(batch_size)
            toks = self.collate_fn([self[idx] for idx in idxs])

            edit_inner = {}
            edit_inner["input_ids"] = toks["inner_input_ids"]
            edit_inner["attention_mask"] = toks["inner_attention_mask"]
            edit_inner["labels"] = toks["inner_labels"]

            loc = {}
            loc["input_ids"] = toks["loc_input_ids"]
            loc["attention_mask"] = toks["loc_attention_mask"]
            loc["labels"] = torch.full((loc["input_ids"].shape[0],), -100)

            edit_outer = {}
            edit_outer["input_ids"] = toks["outer_input_ids"]
            edit_outer["attention_mask"] = toks["outer_attention_mask"]
            edit_outer["labels"] = toks["outer_labels"]

            cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond_")}

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond,
                "pos_pairs": toks["pos_pairs"],
                "pages": toks["pages"],
                "loc_pages": toks["loc_pages"],
                "hard_pos_mask": toks["hard_pos_mask"],
                "hard_neg_mask": toks["hard_neg_mask"]
            }
            assert len(set(batch["pages"])) == len(batch["pages"])
            yield dict_to(batch, self.config.device)


if __name__ == "__main__":
    import types
    import transformers

    cfg = types.SimpleNamespace()
    cfg.data = types.SimpleNamespace()
    cfg.single_batch = False
    cfg.device = "cpu"
    cfg.seed = 0
    tok = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    d = VitC("data/vitaminc", "dev", tok, cfg)
    gen = d.edit_generator(1)
    for idx, b in enumerate(gen):
        if idx > 10:
            break

        print("============================================================================")
        print(tok.batch_decode(b["edit_inner"]["input_ids"], skip_special_tokens=True))
        print("outer", b["edit_outer"]["attention_mask"].max(-1).values.item(), "label", b["edit_outer"]["labels"].item())
        print(tok.batch_decode(b["edit_outer"]["input_ids"], skip_special_tokens=True))
        print("loc", b["loc"]["attention_mask"].max(-1).values.item())
        print(tok.batch_decode(b["loc"]["input_ids"], skip_special_tokens=True))
