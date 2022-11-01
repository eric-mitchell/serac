import datetime
import typing
import numpy as np
import struct
import os
import getpass
import hydra
import logging
import torch
import torch.nn as nn
from collections import defaultdict
import math


LOG = logging.getLogger(__name__)

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()


def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels


def gather_log_probs(logits, labels):
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def off_diagonal(mat):
    assert mat.dim() == 2
    # assert mat.shape[0] == mat.shape[1]

    mask = ~torch.eye(max(mat.shape), dtype=torch.bool)
    mask = mask[:mat.shape[0], :mat.shape[1]]
    off_d = mat[mask]

    assert off_d.numel() == mat.shape[0] * mat.shape[1] - min(mat.shape)

    return off_d


def set_dropout(model, p):
    if p is not None:
        n_reset = 0
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
                n_reset += 1

            if hasattr(m, "dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.dropout, float):
                    m.dropout = p
                    n_reset += 1

            if hasattr(m, "activation_dropout"):  # Requires for BART, which uses F.dropout
                if isinstance(m.activation_dropout, float):
                    m.activation_dropout = p
                    n_reset += 1

        LOG.info(f"Set {n_reset} dropout modules to p={p}")


def _inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]


def shift_targets(config):
    return "t5" not in config.model.name.lower() and "blender" not in config.model.name.lower()


# https://stackoverflow.com/questions/32871539/integer-factorization-in-python
def factorization(n):
    return [(i, n // i) for i in range(1, int(n**0.5) + 1) if n % i == 0]


def scr():
    if os.path.exists("/scr-ssd"):
        scr_dir = "/scr-ssd/" + getpass.getuser()
    else:
        scr_dir = "/scr/" + getpass.getuser()

    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)

    return scr_dir


def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value


def formatted_timestamp(time=None):
    if time is None:
        time = datetime.datetime.now()
    return time.strftime("%d/%m/%Y-%H:%M:%S/%f")


def time_delta_seconds(start, finish=None):
    assert type(start) == str

    t1 = datetime.datetime.strptime(start, "%d/%m/%Y-%H:%M:%S/%f")
    if finish is not None:
        assert type(finish) == str
        t2 = datetime.datetime.strptime(finish, "%d/%m/%Y-%H:%M:%S/%f")
    else:
        t2 = datetime.datetime.now()

    return (t2 - t1).total_seconds()


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict


def safe_backward(loss, parameters, accumulate=1, allow_unused=False, backward=False):
    if backward:
        (loss / accumulate).backward()
    else:
        parameters = list(parameters)  # Capture the generator output
        grads = torch.autograd.grad(loss, parameters, allow_unused=allow_unused)
        nan, inf = False, False
        for g in grads:
            if g is not None:
                nan |= g.isnan().any().item()
                inf |= g.isinf().any().item()

        if not (nan or inf):
            for p, g in zip(parameters, grads):
                if g is None:
                    continue

                if p.grad is None:
                    p.grad = g / accumulate
                else:
                    p.grad += g / accumulate
        else:
            LOG.info(f"Skipping grad accumulation because inf: {inf} nan: {nan}")


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits


def _last_encoder_state(x):
    if hasattr(x, "encoder_last_hidden_state"):
        return x.encoder_last_hidden_state
    else:
        return x.hidden_states[-1]


def load_archive(path):
    import torch

    if not os.path.exists(path):
        # We've not passed an explicit path, but a part of the filename
        wd = hydra.utils.get_original_cwd()
        directories = ["outputs", "multirun"]
        matches = []
        for d in directories:
            search = os.path.join(wd, d)
            for run_dir in os.listdir(search):
                if path in run_dir:
                    matches.append(os.path.join(search, run_dir))
        assert len(matches) == 1, f">1 matches for search {path}; specify exact path"

        full_run_dir = matches[0]
        if "0" in os.listdir(full_run_dir):
            full_run_dir = os.path.join(full_run_dir, "0")
        models_dir = os.path.join(full_run_dir, "models")
        models = os.listdir(models_dir)
        non_bk = [m for m in models if not m.endswith(".bk")]
        assert (
            len(non_bk) == 1
        ), f"Expected a single model in {models_dir}, got {len(non_bk)}"
        path = os.path.join(models_dir, non_bk[0])

    LOG.info(f"Loading checkpoint from {path}")
    archive = torch.load(path, map_location="cpu")
    LOG.info("Load complete.")

    return archive, path


def flatten_dict(d):
    to_process = list(d.items())
    output = {}
    while len(to_process):
        k, v = to_process.pop()
        if isinstance(v, typing.MutableMapping):
            to_process.extend([(f"{k}.{k_}", v_) for (k_, v_) in v.items()])
        else:
            assert k not in output.keys(), "Somehow ended up with duplicate keys"
            output[k] = v

    return output


def add_padding(tokenizer, model):
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)


def add_sep(tokenizer, model):
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.lm_head.weight.data[-1, :] = model.lm_head.weight.data.mean(0)


class EarlyStopper:
    def __init__(self, patience: int, key: str):
        self.best_value = 1e9
        self.best_iter = 0
        self.current_iter = 0
        self.key = key
        self.patience = patience
        self._stop = False

    def update(self, idx, stats):
        assert self.key in stats, f"'{self.key}' not in stats dict"
        value = stats[self.key]
        new_best = value < self.best_value
        if new_best:
            self.best_value = value
            self.best_iter = idx

        self.current_iter = idx
        return new_best

    def should_stop(self):
        self._stop |= self.current_iter - self.best_iter >= self.patience
        return self._stop


class RunningStatAverager:
    def __init__(self, suffix="", exclude=["grad/"], compute_ppl: bool = True):
        self.underlying = None
        self.suffix = suffix
        self.exclude = exclude
        self.compute_ppl = compute_ppl

        self.reset()

    def add(self, d: dict):
        for k, v in d.items():
            if not any([k.startswith(prefix) for prefix in self.exclude]):
                if len(self.suffix):
                    self.underlying[f"{k}_{self.suffix}"].append(v)
                else:
                    self.underlying[k].append(v)

    def average(self):
        average = {}
        for k, v in self.underlying.items():
            if not k.startswith("nll/"):
                average[k] = sum(v) / len(v)
            else:
                assert len(k.split("/")) == 2, f"Invalid key {k}"
                name = k.split("/")[1]
                token_counts = self.underlying[f"n_tokens/{name}"]
                total_nll = sum([nll * c for nll, c in zip(v, token_counts)])
                average[k] = total_nll / sum(token_counts)
                if self.compute_ppl:
                    average[f"perplexity/{name}"] = math.e ** average[k]

        return {k: v if not isinstance(v, torch.Tensor) else v.item() for k, v in average.items()}

    def reset(self):
        self.underlying = defaultdict(list)


class EditBatchSampler:
    def __init__(
        self,
        n,
        memorize_mode=False,
        loc_disjoint=True,
        seed=0,
        hard_neg=False,
        hard_neg_prob=1.0,
        loc_distr_matrix=None,
        loc_idx_matrix=None,
        keep_probs=None,
        mutex=None
    ):
        self.memorize_mode = memorize_mode
        self.n = n
        self.loc_disjoint = loc_disjoint
        self.rng = np.random.default_rng(seed)
        self.hard_neg = hard_neg
        self.hard_neg_prob = hard_neg_prob
        self.loc_probs = loc_distr_matrix
        self.loc_idxs = loc_idx_matrix
        self.keep_probs = np.array(keep_probs)[:self.n] if keep_probs is not None else None
        self.mutex = mutex[:self.n] if mutex is not None else None
        self._init()

    def _init(self):
        idxs = np.arange(self.n)
        if self.keep_probs is not None:
            sample = self.rng.binomial(1, self.keep_probs).astype(np.bool)
            idxs = idxs[sample]

        self.perm = self.rng.permutation(idxs)
        self.edit_position = 0

    def get_edit_idxs(self, batch_size):
        if self.mutex is None:
            idxs = set([int(idx) for idx in self.perm[self.edit_position: self.edit_position + batch_size]])
            self.edit_position += batch_size
        else:
            mutexes = []
            idxs = []

            def notin(x, mutexes):
                for m in mutexes:
                    if x in m or m in x:
                        return False
                return True
            while len(idxs) < batch_size:
                new_idx = self.perm[self.edit_position]
                if notin(self.mutex[new_idx], mutexes):
                    mutexes.append(self.mutex[new_idx])
                    idxs.append(int(new_idx))
                self.edit_position += 1
                if self.edit_position == self.perm.shape[0]:
                    return None

            idxs = set(idxs)

        return idxs

    def sample(self, batch_size, return_hard_flag=False):
        if self.memorize_mode:
            return list(range(batch_size)), list(range(batch_size, batch_size * 2))

        if self.edit_position + batch_size >= self.perm.shape[0]:
            self._init()  # Re-start if we end with a partially-sized batch

        edit_idxs = self.get_edit_idxs(batch_size)
        if edit_idxs is None:
            self._init()
            edit_idxs = self.get_edit_idxs(batch_size)
            if edit_idxs is None:
                raise RuntimeError(f"No valid batches of size {batch_size} exist!")

        if self.hard_neg:
            assert self.loc_probs is not None, "hard_neg is on, but don't have distance matrix!"

        def get_loc_idxs():
            if self.hard_neg and self.rng.uniform() < self.hard_neg_prob:
                return [int(self.rng.choice(self.loc_idxs[idx], p=self.loc_probs[idx])) for idx in edit_idxs], True
            else:
                # Use deterministic implementation in case edit batches are large
                non_edit_idxs = list(set(range(self.n)) - set(edit_idxs))
                return [int(idx) for idx in self.rng.choice(non_edit_idxs, batch_size)], False

        loc_idxs, hard = get_loc_idxs()
        if self.loc_disjoint:
            steps = 0
            while len(edit_idxs.intersection(set(loc_idxs))) > 0:
                loc_idxs, hard = get_loc_idxs()
                steps += 1
                if steps > 100:
                    raise RuntimeError("Can't find disjoint loc_idxs and edit_idxs!")

        if return_hard_flag:
            return list(edit_idxs), loc_idxs, hard
        else:
            return list(edit_idxs), loc_idxs


def parent_module(model, pname):
    comps = pname.split('.')
    parent = model
    for comp in comps[:-1]:
        if hasattr(parent, comp):
            parent = getattr(parent, comp)
        elif comp.isdigit():
            parent = parent[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    assert hasattr(parent, comps[-1])
    return parent


def build_distr_matrix(edit_qs, config, loc_qs=None, slice_size=1000):
    n = len(edit_qs)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_neighbors = config.data.hard_neg_neighbors
    num_exclude = config.data.hard_neg_exclude
    temp = config.data.hard_neg_temp

    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import pytorch_cos_sim
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=scr()).to(device)

    ind_matrix = torch.zeros((n, num_neighbors - num_exclude), dtype=torch.long)
    distr_matrix = torch.full((n, num_neighbors - num_exclude), float('nan'))
    edit_encodings = torch.FloatTensor(embedding_model.encode(edit_qs, batch_size=256)).to(device)

    # If loc_qs is None then build the similarity matrix between edit_qs and itself
    loc_encodings = edit_encodings if loc_qs is None else embedding_model.encode(loc_qs, batch_size=256)
    if isinstance(loc_encodings, np.ndarray):
        loc_encodings = torch.FloatTensor(loc_encodings).to(device)

    for idx in range(0, n, slice_size):
        end_idx = idx + slice_size if idx + slice_size <= n else n
        slice_encodings = edit_encodings[idx:end_idx]
        sim_rows = pytorch_cos_sim(slice_encodings, loc_encodings)
        indices = sim_rows.topk(num_neighbors, -1).indices[:, num_exclude:]
        ind_matrix[idx:end_idx] = indices.cpu()
        distr_matrix[idx:end_idx] = sim_rows.gather(-1, indices).div(temp).exp().cpu()

    assert not torch.isnan(distr_matrix).any()

    LOG.info(f"Built hard negative distribution matrix of size {distr_matrix.shape}")
    distr_matrix = distr_matrix.numpy()
    distr_matrix = distr_matrix / distr_matrix.sum(-1, keepdims=True)
    return distr_matrix, ind_matrix.numpy()

