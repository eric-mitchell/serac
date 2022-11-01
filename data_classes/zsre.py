import jsonlines
import json
import itertools
from torch.utils.data import Dataset
import random
from utils import EditBatchSampler, dict_to, build_distr_matrix
import torch
from transformers import BartTokenizerFast, BartTokenizer
import logging
import numpy as np
import re

LOG = logging.getLogger(__name__)


YN_PROMPTS = [
    "True or False: ",
    "True/False: ",
    "T/F: ",
    "Answer True or False: ",
]


def prompt_yn(q):
    prompt = random.choice(YN_PROMPTS)
    if np.random.uniform() < 0.5:
        prompt = prompt.lower()
    return prompt + q


class Seq2SeqAugmentedKILT(Dataset):
    def __init__(
        self,
        split,
        tokenizer,
        config,
        max_length=32,
        return_view=False,
        all_views=False,
    ):
        super().__init__()
        self.tok = tokenizer
        self.config = config

        assert split in ["train", "dev", "test"]
        data_path = config.data.zsre_path.format(split)

        if config.data.zsre_impl:
            with open(config.data.zsre_impl_path.format(split)) as f:
                impls = iter(json.load(f))
        else:
            impls = itertools.cycle([[]])

        if config.data.zsre_yn:
            with open(config.data.zsre_yn_path.format(split)) as f:
                yns = iter(list(f))
        else:
            yns = itertools.cycle([''])

        if config.data.zsre_eval_idxs is not None:
            with open(config.data.zsre_eval_idxs, "r") as f:
                eval_idxs = set([int(l) for l in f])
        else:
            eval_idxs = None

        def extract(d):
            ex = {k: d[k] for k in ["input", "prediction", "alternatives", "filtered_rephrases", "output"]}
            if ex["input"] in ex["filtered_rephrases"]:
                ex["filtered_rephrases"].remove(ex["input"])
            return ex

        def filter_(d):
            if "sex" in d["input"] or "gender" in d["input"]:
                return False
            return True

        self.impls = []
        self.all_impls = []
        self.data = []
        self.all_data = []
        self.yn = []
        self.all_yn = []
        empty_yn = 0
        with jsonlines.open(data_path) as f:
            for idx, d in enumerate(f):
                try:
                    impl_set = next(impls)
                except StopIteration:
                    impl_set = []
                try:
                    yn_q = next(yns)
                except StopIteration:
                    yn_q = ''
                extracted = extract(d)
                self.all_data.append(extracted)
                self.all_impls.append(impl_set)
                self.all_yn.append(yn_q)
                if len(extracted["alternatives"]) > 0 and len(extracted["filtered_rephrases"]) > 0:
                    if eval_idxs is None or idx in eval_idxs:
                        if filter_(extracted):
                            self.data.append(extracted)
                            self.impls.append(impl_set)
                            self.yn.append(yn_q)
                            if len(yn_q) == 0:
                                empty_yn += 1
        LOG.info(f"Empty {split} yn questions: {empty_yn}")

        self.max_length = max_length
        self.all_views = all_views
        self.return_view = return_view
        if self.config.data.zsre_nq:
            self.use_nq = True
            LOG.info("** Using natural questions for zsre base samples **")
            from data_classes.nq import NQDataset
            self.nq = NQDataset(
                self.config.data.nq_path + ("/train.json" if "train" in data_path else "/validation.json"),
                tokenizer,
                config,
            )
        else:
            self.use_nq = False

            divisor = 2 + int(self.config.data.zsre_impl) + int(self.config.data.zsre_yn)
            n_per_dist = len(self.data) // divisor
            remain = len(self.data) - n_per_dist * divisor
            self.loc_data = []
            base_data = [(sample["input"], sample["output"][0]["answer"]) for sample in self.data]
            random.shuffle(base_data)
            self.loc_data += base_data[:n_per_dist + remain]

            rephrase_data = [(r, sample["output"][0]["answer"]) for sample in self.data for r in sample["filtered_rephrases"]]
            random.shuffle(rephrase_data)
            self.loc_data += rephrase_data[:n_per_dist]

            if self.config.data.zsre_impl:
                impl_data = [(q, a) for impl_set in self.impls for (q, a, _) in impl_set]
                random.shuffle(impl_data)
                self.loc_data += impl_data[:n_per_dist]

            if self.config.data.zsre_yn:
                yn_data = [(prompt_yn(y), ("True" if np.random.uniform() < 0.5 else "False")) for y in self.yn]
                random.shuffle(yn_data)
                self.loc_data += yn_data[:n_per_dist]

            LOG.info(f"Data size {len(self.data)}; loc data size {len(self.loc_data)}")

        self.loc_distr_matrix, self.loc_idx_matrix = None, None
        if self.config.data.hard_neg:
            edit_qs = [sample["input"] for sample in self.data]
            loc_qs = self.nq.questions if self.use_nq else [d[0] for d in self.loc_data]
            self.loc_distr_matrix, self.loc_idx_matrix = build_distr_matrix(edit_qs, config, loc_qs=loc_qs)

    def is_bart(self):
        return isinstance(self.tok, BartTokenizer) or isinstance(self.tok, BartTokenizerFast)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item, seed=None):
        orig_label = self.data[item]["output"][0]["answer"]
        escaped_orig_label = re.escape(orig_label)
        impls = self.impls[item]
        if len(impls):
            impls = [i for i in self.impls[item] if orig_label in i[0]]

        impl = random.choice(impls) if len(impls) else None

        new_label = random.choice(self.data[item]["alternatives"])
        if impl is not None and len(re.findall(escaped_orig_label, impl[0], flags=re.IGNORECASE)) > 0:
            implq, impla, _ = impl
            implq = re.sub(escaped_orig_label, new_label, implq, flags=re.IGNORECASE)
        else:
            implq = impla = None

        yn = self.yn[item].strip()
        yn = yn.replace(' - ', '-')
        if len(yn) > 0 and len(re.findall(escaped_orig_label, yn, flags=re.IGNORECASE)) > 0:
            if random.uniform(0, 1) < 0.5 or len(self.data[item]["alternatives"]) == 1:
                ynq = re.sub(escaped_orig_label, new_label, yn, flags=re.IGNORECASE)
                yna = "True"
            else:
                yn_alt_label = random.choice(self.data[item]["alternatives"])
                while yn_alt_label == new_label:
                    yn_alt_label = random.choice(self.data[item]["alternatives"])

                ynq = re.sub(escaped_orig_label, yn_alt_label, yn, flags=re.IGNORECASE)
                yna = "False"

            ynq = prompt_yn(ynq)
        else:
            ynq = yna = None

        main_input = self.data[item]["input"]
        rephrase = random.choice(self.data[item]["filtered_rephrases"])

        MAIN, REPHRASE, IMPL, YN = range(4)
        impl_prob = float((impla is not None) and self.config.data.zsre_impl)
        yn_prob = float((yna is not None) and self.config.data.zsre_yn)
        probs = np.array([1., 1., impl_prob, yn_prob])
        outer_type = np.random.choice([MAIN, REPHRASE, IMPL, YN], p=probs / probs.sum())

        if outer_type == MAIN:
            outer_q, outer_a = main_input, new_label
        elif outer_type == REPHRASE:
            outer_q, outer_a = rephrase, new_label
        elif outer_type == IMPL:
            outer_q, outer_a = implq, impla
        elif outer_type == YN:
            outer_q, outer_a = ynq, yna
        else:
            raise RuntimeError

        output = {
            "src": main_input,
            "pred": self.data[item]["prediction"],
            "rephrase": rephrase,
            "alt": new_label,
            "outer_q": outer_q,
            "outer_a": outer_a,
            "answers": [x["answer"] for x in self.data[item]["output"]],
            "cond": "{} >> {} || {}".format(
                self.data[item]["prediction"],
                new_label,
                self.data[item]["input"],
            ),
            "hard": outer_type == IMPL or outer_type == YN
        }

        return output

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        ne = len(src) // 2  # self.config.data.n_edits
        trg = (
            [b["answers"][0] for b in batch[:-ne]] +
            [b["alt"] for b in batch[-ne:]]
        )

        batches = {
            f"{k1}_{k2}": v2
            for k1, v1 in {
                "src": src,
                "trg": trg,
                "cond": [b["cond"] for b in batch[-ne:]],
                "outer_q": [b["outer_q"] for b in batch[-ne:]],
                "outer_a": [b["outer_a"] for b in batch[-ne:]],
            }.items()
            for k2, v2 in self.tok(
                v1,
                return_tensors="pt",
                padding=True,
                max_length=self.max_length,
                truncation=True,
            ).items()
        }

        if self.is_bart():  # For consistency with de Cao et al
            batches["trg_input_ids"][:, 0] = self.tok.eos_token_id
            batches["outer_a_input_ids"][:, 0] = self.tok.eos_token_id
        batches["raw"] = batch
        batches["hard_pos_mask"] = [b["hard"] for b in batch]
        return batches

    def _check_padding(self, ids):
        if (ids[:, 0] == self.tok.pad_token_id).any():
            raise ValueError("Left-padding not supported")

    def mask_padding_for_labels(self, labels):
        return labels.masked_fill(labels == self.tok.pad_token_id, -100)

    def edit_generator(self, batch_size, n=None):
        if n is None:
            n = len(self)
        sampler = EditBatchSampler(
            n,
            memorize_mode=self.config.single_batch,
            loc_disjoint=not self.use_nq,
            seed=self.config.seed,
            hard_neg=self.config.data.hard_neg,
            hard_neg_prob=self.config.data.hard_neg_prob,
            loc_distr_matrix=self.loc_distr_matrix,
            loc_idx_matrix=self.loc_idx_matrix,
        )

        while True:
            edit_idxs, loc_idxs, hard_neg_flag = sampler.sample(batch_size, return_hard_flag=True)

            idxs = loc_idxs + edit_idxs
            toks = self.collate_fn([self[idx] for idx in idxs])

            # ne = self.config.data.n_edits
            ne = batch_size

            edit_inner = {}
            edit_inner["input_ids"] = toks["src_input_ids"][-ne:]
            edit_inner["attention_mask"] = toks["src_attention_mask"][-ne:]
            if self.is_bart():
                edit_inner["decoder_input_ids"] = toks["trg_input_ids"][-ne:]
                edit_inner["decoder_attention_mask"] = toks["trg_attention_mask"][-ne:]
            edit_inner["labels"] = self.mask_padding_for_labels(toks["trg_input_ids"][-ne:])

            if self.config.data.rephrase:
                edit_outer = {}
                edit_outer["input_ids"] = toks["outer_q_input_ids"]
                edit_outer["attention_mask"] = toks["outer_q_attention_mask"]
                if self.is_bart():
                    edit_outer["decoder_input_ids"] = toks["outer_a_input_ids"]
                    edit_outer["decoder_attention_mask"] = toks["outer_a_attention_mask"]
                edit_outer["labels"] = self.mask_padding_for_labels(toks["outer_a_input_ids"])
            else:
                edit_outer = edit_inner

            loc = {}
            if self.use_nq:
                batch = [self.nq[idx] for idx in loc_idxs]
            else:
                batch = [self.loc_data[idx] for idx in loc_idxs]
            questions = [b[0] for b in batch]
            answers = [b[1] for b in batch]
            loc = dict(self.tok(questions, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True))
            trg_toks = dict(self.tok(answers, return_tensors="pt", padding=True, max_length=self.max_length, truncation=True))
            if self.is_bart():
                trg_toks["input_ids"][:, 0] = self.tok.eos_token_id
                loc["decoder_input_ids"] = trg_toks["input_ids"]
            loc["decoder_attention_mask"] = trg_toks["attention_mask"]
            loc["labels"] = self.mask_padding_for_labels(trg_toks["input_ids"])

            cond = {k[5:]: v for k, v in toks.items() if k.startswith("cond")}

            if self.config.data.flip_inner_outer and np.random.uniform() < 0.5:
                edit_inner, edit_outer = edit_outer, edit_inner

            pos_pairs = torch.arange(batch_size, device=self.config.device).unsqueeze(-1).repeat(1, 2)
            assert edit_outer["input_ids"].shape[0] == batch_size

            batch = {
                "edit_inner": edit_inner,
                "edit_outer": edit_outer,
                "loc": loc,
                "cond": cond,
                "raw": toks["raw"],
                "pos_pairs": pos_pairs,
                "hard_pos_mask": toks["hard_pos_mask"][-ne:],
                "hard_neg_mask": [hard_neg_flag] * loc["input_ids"].shape[0]
            }

            yield dict_to(batch, self.config.device)


def default_dataset(split="val"):
    import transformers
    from types import SimpleNamespace
    import numpy as np
    config = SimpleNamespace()
    config.device = "cpu"
    config.single_batch = False
    config.data = SimpleNamespace()
    config.data.rephrase = True
    config.data.zsre_path = "data/zsre/structured_zeroshot-{}-new_annotated_final.jsonl"
    config.data.zsre_nq = True
    config.data.zsre_impl = True
    config.data.zsre_impl_path = "data/zsre/impl_{}.json"
    config.data.zsre_yn = True
    config.data.zsre_yn_path = "data/zsre/zsre_yn_{}.txt"
    config.data.nq_path = "data/nq"
    config.data.zsre_eval_idxs = None  # "data/zsre/good_impl_eval_idxs.txt"
    config.data.flip_inner_outer = False
    config.batch_size = 100
    config.val_batch_size = 20
    config.data.hard_neg = False
    config.data.hard_neg_neighbors = 20
    config.data.hard_neg_exclude = 0
    config.data.hard_neg_temp = 0.1
    config.data.hard_neg_prob = 0.5
    config.single_batch = False
    config.seed = 0
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    split = (
        "data/zsre/structured_zeroshot-train-new_annotated_final.jsonl" if split == "train"
        else "data/zsre/structured_zeroshot-dev-new_annotated_final.jsonl" if split == "val"
        else "data/zsre/structured_zeroshot-test-new_annotated_final.jsonl"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained('google/t5-small-ssm-nq')
    return config, tokenizer, Seq2SeqAugmentedKILT("dev", tokenizer, config)


if __name__ == '__main__':
    config, tokenizer, dataset = default_dataset()
    batch = next(dataset.edit_generator(config.batch_size))
    import pdb; pdb.set_trace()
    for idx in range(300):
        batch = next(gen)
        edit_in = tokenizer.decode(batch["edit_inner"]["input_ids"][0])
        edit_out = tokenizer.decode(batch["edit_outer"]["input_ids"][0])
        labs_in = batch["edit_inner"]["labels"][0]
        labs_out = batch["edit_outer"]["labels"][0]
        edit_in_labels = tokenizer.decode(labs_in[labs_in != -100])
        edit_out_labels = tokenizer.decode(labs_out[labs_out != -100])
        loc = tokenizer.decode(batch["loc"]["input_ids"][0])
        loc_labs = batch["loc"]["labels"][0]
        loc_labs = tokenizer.decode(loc_labs[loc_labs != -100])
        print("[e_i]" + edit_in + " || " + edit_in_labels)
        print("[e_o]" + edit_out + " || " + edit_out_labels)
        print("[loc]" + loc + " || " + loc_labs)