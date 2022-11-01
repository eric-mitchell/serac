import logging
import os
import shutil
import tempfile
import time
import json

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

import wandb

from metrics import retain_rate
from losses import kl_loc_loss, balanced_bce
import utils
from utils import (_logits, safe_backward, RunningStatAverager, EarlyStopper,
                   formatted_timestamp, time_delta_seconds, off_diagonal)


LOG = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        self.model = model
        self.config = config
        self.early_stop_key = self.config.early_stop_key
        if config.train_base:
            self.original_model = self.model.model_constructor()
            self.original_model.load_state_dict(self.model.model.state_dict())
            self.original_model.to(self.config.device)
        else:
            self.original_model = self.model.model

        self.model.to(self.config.device)

        self.train_set = train_set
        self.val_set = val_set

        if self.config.eval_only:
            # Eval once and quit
            self.config.max_iters = 0

        if not self.config.eval_only:
            self.opt = getattr(torch.optim, config.opt)(self.model.outer_parameters(grouped=True), lr=config.lr)
            LOG.info(f"Built optimizer {self.opt}")
        if config.archive is not None:
            archive, config.archive = utils.load_archive(str(config.archive))
            LOG.info("WHY DO WE HAVE TO DO THIS NOW?")
            if "model_config" in archive["model"]:
                archive["model"]["model_config"].torch_dtype = str(archive["model"]["model_config"].torch_dtype)
            self.model.load_state_dict(archive["model"])
            del archive["model"]
            if not self.config.eval_only:
                self.opt.load_state_dict(archive["opt"])
            del archive["opt"]

            self.archive = archive  # Save for later to load e.g. lr_opt params if they exist
        else:
            self.archive = None

        # outfiles
        with open(os.getcwd() + "/config.json", "w") as f:
            json.dump(OmegaConf.to_container(config), f)

        model_dir = os.path.join(os.getcwd(), 'models')
        if not (self.config.debug and not self.config.save):
            os.makedirs(model_dir)
        run_date = os.getcwd().split('/')[-1]
        self.run_date = run_date
        safe_model_name = self.config.model.name.split("/")[-1]  # Make sure no slashes
        self.save_path = f"{model_dir}/{safe_model_name}.{run_date}"

        if not (self.config.debug or self.config.eval_only):
            wandb_dir = tempfile.mkdtemp()
            wandb_name = f"{self.config.dataset} - {self.config.alg} - {safe_model_name} - {run_date}"
            if self.config.ref is not None:
                wandb_name += f" - {self.config.ref}"
            LOG.info(f"Writing wandb run \"{wandb_name}\" to {wandb_dir}")
            wandb.init(
                project="serac",
                config=utils.flatten_dict(self.config),
                name=wandb_name,
                dir=wandb_dir,
                tags=[self.config.ref] if self.config.ref is not None else None
            )

        self.start_time = formatted_timestamp()

    def save_state(self, stats):
        if (self.config.debug and not self.config.save) or self.config.eval_only:
            return

        obj = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "val_stats": stats,
            "start_time": self.start_time,
            "elapsed_time": time_delta_seconds(self.start_time),
            "step": self.global_iter
        }
        LOG.info(f"Saving model to {self.save_path}")

        if os.path.exists(self.save_path):
            bk_path = f"{self.save_path}.bk"
            LOG.info(f"Moving old archive to {bk_path}")
            os.rename(self.save_path, bk_path)

        torch.save(obj, self.save_path)
        LOG.info("Write complete.")

    def echo(self, train_step, info_dict, pretty=False):
        if not self.config.silent:
            sep = "\n" if pretty else "; "

            def key_format(k):
                return k.ljust(20) if pretty else k
            LOG.info(f"Step {train_step}:")
            LOG.info(sep.join([f"{key_format(k)}: {v: 0.5f}" for k, v in info_dict.items()]))

    def wandb_log(self, step, info_dict):
        if not (self.config.debug or self.config.eval_only):
            wandb.log(info_dict, step=step)

    def run(self):
        averager = RunningStatAverager("train")
        stopper = EarlyStopper(self.config.early_stop_patience, self.early_stop_key)
        self.global_iter = 0
        for global_iter in range(0, self.config.max_iters):
            self.global_iter = global_iter

            if not self.config.eval_only:
                train_info = self.train_step()
                averager.add(train_info)

                if global_iter % self.config.log_interval == 0:
                    avg_info = averager.average()
                    averager.reset()
                    self.echo(global_iter, avg_info)
                    self.wandb_log(global_iter, avg_info)

            if global_iter % self.config.val_interval == 0:
                val_info = self.validate(steps=self.config.val_steps)
                self.echo(global_iter, val_info)
                self.wandb_log(global_iter, val_info)

                if stopper.update(self.global_iter, val_info):
                    self.save_state(val_info)  # New best

                if stopper.should_stop():
                    LOG.info(f"No decrease in {self.config.early_stop_key} for {self.config.early_stop_patience} steps")
                    break

        if not self.config.eval_only:
            LOG.info(f"Training complete after {self.global_iter+1} steps.")

        if not self.config.eval.final_eval:
            return

        if not self.config.eval_only:
            if (not self.config.debug) or self.config.save:
                archive = torch.load(self.save_path, map_location="cpu")
                LOG.info(f"Loading best model from step {archive['step']}, elapsed time {archive['elapsed_time']}")
                self.model.to("cpu")
                self.model.load_state_dict(archive["model"])
                self.model.to(self.config.device)

        val_steps = 200 if self.config.debug else None
        val_info = self.validate(log=True, steps=val_steps)
        self.echo(self.global_iter, val_info, pretty=True)
        self.wandb_log(self.global_iter + self.config.val_interval, val_info)

        if self.config.results_dir is not None:
            results_path = f"{self.config.results_dir}/results_{self.run_date}.json"
            latest_path = f"{self.config.results_dir}/results_latest.json"
        else:
            results_path = f"{os.getcwd()}/results.json"
            latest_path = f"{os.getcwd()}/results_latest.json"

        with open(results_path, "w") as f:
            json.dump({"results": val_info, "config": OmegaConf.to_container(self.config)}, f)
            LOG.info("Wrote results to:")
            LOG.info(results_path)

        shutil.copy(results_path, latest_path)
        LOG.info("Copied to:")
        LOG.info(latest_path)


class EditTrainer(BaseTrainer):
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        super().__init__(model, config, train_set, val_set)

        self.edit_gen = self.train_set.edit_generator(batch_size=config.batch_size)

        if hasattr(self.config, "ft"):
            if getattr(self.config.ft, "use_locality", False):
                batch = next(self.edit_gen)
                self.model.loc_ids = batch["loc"]["input_ids"]
                self.model.loc_masks = batch["loc"]["attention_mask"]

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_logits = self.model(**batch["loc"])

        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start
        edited_model.train(training)

        info_dict = {}
        with torch.set_grad_enabled(training):
            # Editing loss
            pos_pairs = batch["pos_pairs"]
            if self.config.data.n_outer_max is not None:
                # Truncate to keep memory consumption reasonable for many edits
                for k, v in batch["edit_outer"].items():
                    batch["edit_outer"][k] = v[:self.config.data.n_outer_max]

                pos_pairs_trunc_idxs = torch.where(pos_pairs[:, 0] < self.config.data.n_outer_max)[0]
                pos_pairs = pos_pairs[pos_pairs_trunc_idxs]

            HAS_OUTER_DATA = pos_pairs.numel() > 0
            if HAS_OUTER_DATA:
                post_edit_logits = edited_model(**batch["edit_outer"])
                if self.config.task == "sent":
                    with torch.no_grad():
                        kwargs = dict(
                            pre_edit_logits=self.model(**batch["edit_outer"]),
                            post_edit_logits=post_edit_logits.detach(),
                            inner_sent=batch["inner_sent"],
                            outer_sent=batch["outer_sent"],
                            unlikelihood=self.config.unlikelihood,
                        )
                else:
                    kwargs = {}

                post_edit_dict = self.model.edit_loss_fn(
                    post_edit_logits,
                    batch["edit_outer"]["labels"],
                    **kwargs,
                )
                l_edit = post_edit_dict["nll"]
            else:
                l_edit = torch.tensor(0.0)

            # Locality loss
            post_base_logits = edited_model(**batch["loc"])
            kl_mask = batch["loc"].get("decoder_attention_mask", batch["loc"]["attention_mask"])
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc

        if training:
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs)

        info_dict['loss/edit'] = l_edit.item()
        info_dict['loss/loc'] = l_loc.item()
        info_dict["kl/edit"] = l_loc.item()
        if HAS_OUTER_DATA:
            info_dict['edit/acc'] = post_edit_dict["acc"].item()
            info_dict['edit/log_prob'] = post_edit_dict["log_prob"].item()
            info_dict['edit/prob'] = post_edit_dict["prob"].item()

        info_dict["retain/edit"] = retain_rate(base_logits, post_base_logits, batch["loc"]["labels"] != -100)
        info_dict["time/edit"] = edit_time

        if HAS_OUTER_DATA:
            if self.config.task == "sent":
                info_dict["edit/acc_sent"] = post_edit_dict["acc_sent"].item()
            for k, v in post_edit_dict.items():
                if isinstance(v, torch.Tensor):
                    info_dict[f"stat_dump/{k}"] = v.item()
                else:
                    info_dict[f"stat_dump/{k}"] = v

        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_base_logits = _logits(self.original_model(**batch["loc"]))

            base_logits = self.model(**batch["loc"])
            l_base = kl_loc_loss(original_base_logits.detach(), base_logits, mask=kl_mask)

            if training:
                safe_backward(l_base, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True)

            info_dict['loss/base'] = l_base.item()
            info_dict["retain/orig_pre"] = retain_rate(original_base_logits, base_logits.detach(), batch["loc"]["labels"] != -100)
            info_dict["retain/orig_post"] = retain_rate(original_base_logits, post_base_logits, batch["loc"]["labels"] != -100)
            info_dict["kl/orig_post"] = kl_loc_loss(original_base_logits.detach(), post_base_logits, mask=kl_mask.detach()).item()
        else:
            l_base = torch.tensor(0.)

        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(next(self.edit_gen), training=True)

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(self.model.outer_parameters(), self.config.grad_clip,
                                                  error_if_nonfinite=True)
            info_dict['grad'] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if hasattr(self.model, "edit_lrs"):
                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f'lr/lr{lr_idx}'] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        if self.config.task in ["fc"]:
            draw_pre = f"{stats['acc/pre_val']:<12.5f}"
            draw_post = f"{stats['acc/post_val']:<12.5f}"
            draw_diff = f"{stats['acc/pre_val']-stats['acc/post_val']:<12.5f}"
            dn = "acc"  # drawdown name
        elif self.config.task in ["sent"]:
            acc = f"{stats['edit/acc_sent_val']:<12.5f}"
            draw_pre = ""
            draw_post = ""
            if self.config.alg == "enn":
                draw_diff = f"{stats['kl/orig_post_val']:<12.5f}"
            else:
                draw_diff = f"{stats['kl/edit_val']:<12.5f}"
            dn = "kl"
        elif self.config.task.endswith("nli") or self.config.task in ["qa"]:
            draw_pre = ""
            draw_post = ""
            if self.config.alg == "enn":
                draw_diff = f"{stats['retain/orig_post_val']:<12.5f}"
            else:
                draw_diff = f"{stats['retain/edit_val']:<12.5f}"
            dn = "retain"
        else:
            raise RuntimeError(f"Didn't recognize task {self.config.task}")

        LOG.info(f"Batch {prog} edit: {acc} {dn}_pre: {draw_pre} {dn}_post: {draw_post} {dn}_delta: {draw_diff} it_time: {elapsed:.4f}")

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        n_batches = steps // self.config.val_batch_size
        if log:
            LOG.info(f"Beginning evaluation for {n_batches} batches...")
        averager = RunningStatAverager("val")
        val_edit_gen = (
            self.val_set.edit_generator(batch_size=self.config.val_batch_size, n=steps,
                                        do_sample=self.config.data.sent_eval_sample)
            if self.config.task == "sent"
            else self.val_set.edit_generator(batch_size=self.config.val_batch_size, n=steps)
        )

        start_time = time.time()
        for val_batch_idx in range(n_batches):
            _, _, _, _, info_dict = self.edit_step(next(val_edit_gen), training=False)
            averager.add(info_dict)

            if log and self.config.eval.verbose and (val_batch_idx + 1) % self.config.eval.log_interval == 0:
                self._inline_validation_log(val_batch_idx, averager.average(), start_time, n_batches)

        if log and self.config.eval.verbose:
            self._inline_validation_log(val_batch_idx, averager.average(), start_time, n_batches)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / n_batches

        return stats


class SupervisedTrainer(EditTrainer):
    def __init__(self, model, config, train_set: Dataset, val_set: Dataset):
        super().__init__(model, config, train_set, val_set)
        self.early_stop_key = "loss/total_supervised_val"

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        # Do the edit
        start = time.time()
        with torch.no_grad():
            base_logits = self.model(**batch["loc"])

        edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
        edit_time = time.time() - start
        edited_model.train(training)

        info_dict = {}
        with torch.set_grad_enabled(training):
            # Editing loss

            pos_pairs = batch["pos_pairs"]
            HAS_OUTER_DATA = pos_pairs.numel() > 0
            if self.config.data.n_outer_max is not None:
                # Truncate to keep memory consumption reasonable for many edits
                for k, v in batch["edit_outer"].items():
                    batch["edit_outer"][k] = v[:self.config.data.n_outer_max]

                if HAS_OUTER_DATA:
                    pos_pairs_trunc_idxs = torch.where(pos_pairs[:, 0] < self.config.data.n_outer_max)[0]
                    pos_pairs = pos_pairs[pos_pairs_trunc_idxs]

            if HAS_OUTER_DATA:
                post_edit_logits, edit_cls_logits, post_cntr_logits, edit_model_stats = edited_model(**batch["edit_outer"],
                                                                                                     return_logits_only=False,
                                                                                                     pos_pairs=pos_pairs)
                if self.config.task == "sent":
                    with torch.no_grad():
                        kwargs = dict(
                            pre_edit_logits=self.model(**batch["edit_outer"]),
                            post_edit_logits=post_edit_logits.detach(),
                            inner_sent=batch["inner_sent"],
                            outer_sent=batch["outer_sent"],
                            unlikelihood=self.config.unlikelihood,
                        )
                else:
                    kwargs = {}

                # Need to do these two evals separately, once with the counterfactual model logits,
                #  once with the mixed logits (to get the overall model scores)
                post_cntr_dict = self.model.edit_loss_fn(
                    post_cntr_logits,
                    batch["edit_outer"]["labels"],
                    **kwargs,
                )

                with torch.no_grad():
                    post_edit_dict = self.model.edit_loss_fn(
                        post_edit_logits,
                        batch["edit_outer"]["labels"],
                        **kwargs,
                    )

                l_cntr = post_cntr_dict["nll"]
            else:
                l_cntr = torch.tensor(0.0)
                edit_model_stats = {}

            edit_model_stats = {f"{k}_edit": v for k, v in edit_model_stats.items()}
            info_dict = {**info_dict, **edit_model_stats}

            # Locality loss
            post_base_logits, loc_cls_logits, _, loc_model_stats = edited_model(**batch["loc"], return_logits_only=False)

            loc_model_stats = {f"{k}_loc": v for k, v in loc_model_stats.items()}
            info_dict = {**info_dict, **loc_model_stats}

            # Used to be: cls_pos_logits = edit_cls_logits.diag()[pos_mask]
            if HAS_OUTER_DATA:
                cls_pos_logits = edit_cls_logits[pos_pairs[:, 0], pos_pairs[:, 1]]
                cls_pos_labels = torch.ones_like(cls_pos_logits)

            cls_neg_logits = loc_cls_logits
            if self.config.rep.use_all_negatives:
                cls_neg_logits = loc_cls_logits.view(-1)
            else:
                cls_neg_logits = loc_cls_logits.diag()
            cls_neg_labels = torch.zeros_like(cls_neg_logits)
            if HAS_OUTER_DATA:
                cls_logits = torch.cat((cls_pos_logits, cls_neg_logits))
                cls_labels = torch.cat((cls_pos_labels, cls_neg_labels))
            else:
                cls_logits = cls_neg_logits
                cls_labels = cls_neg_labels

            # in this case, the classifier loss is our "locality" loss
            l_cls = balanced_bce(cls_logits, cls_labels)

            if self.config.log_errors:
                if HAS_OUTER_DATA:
                    pos_acc = (cls_pos_logits[0].exp() > 0.5) == cls_pos_labels[0]
                else:
                    pos_acc = torch.tensor(True)
                neg_acc = (cls_neg_logits[0].exp() > 0.5) == cls_neg_labels[0]
                if (not pos_acc) or (not neg_acc):
                    LOG.info("*" * 40)

                    def valid(x):
                        return x[x != -100]
                    LOG.info(self.train_set.tok.decode(batch["edit_inner"]["input_ids"][0], skip_special_tokens=True))
                    LOG.info(self.train_set.tok.decode(valid(batch["edit_inner"]["labels"][0]), skip_special_tokens=True))
                    LOG.info(self.train_set.tok.decode(batch["edit_outer"]["input_ids"][0], skip_special_tokens=True))
                    LOG.info(self.train_set.tok.decode(valid(batch["edit_outer"]["labels"][0]), skip_special_tokens=True))
                    LOG.info(self.train_set.tok.decode(batch["loc"]["input_ids"][0], skip_special_tokens=True))
                    LOG.info(self.train_set.tok.decode(valid(batch["loc"]["labels"][0]), skip_special_tokens=True))
                    LOG.info("Pos acc: " + str(pos_acc.long().item()))
                    LOG.info("Neg acc: " + str(neg_acc.long().item()))

        l_total_sup = self.config.cedit * l_cntr + self.config.cloc * l_cls

        if training:
            safe_backward(l_total_sup, self.model.outer_parameters(), self.config.accumulate_bs,
                          allow_unused=not HAS_OUTER_DATA, backward=self.config.rep.checkpoint_grad)

        # Log all kinds of stuff now
        cls_n_acc = ((cls_neg_logits.exp() > 0.5) == cls_neg_labels).float().mean().item()
        if HAS_OUTER_DATA:
            cls_p_acc = ((cls_pos_logits.exp() > 0.5) == cls_pos_labels).float().mean().item()
            info_dict['cls/acc'] = (cls_p_acc + cls_n_acc) / 2
            info_dict['cls/pos_acc'] = cls_p_acc
        else:
            info_dict['cls/acc'] = cls_n_acc

        info_dict['cls/neg_acc'] = cls_n_acc
        info_dict['loss/cntr'] = l_cntr.item()
        info_dict['loss/cls'] = l_cls.item()
        kl_mask = batch["loc"].get("decoder_attention_mask", batch["loc"]["attention_mask"])
        info_dict["retain/edit"] = retain_rate(base_logits, post_base_logits, batch["loc"]["labels"] != -100)
        info_dict["kl/edit"] = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask).item()
        if HAS_OUTER_DATA:
            info_dict['cntr/acc'] = post_cntr_dict["acc"].item()
            info_dict['cntr/log_prob'] = post_cntr_dict["log_prob"].item()
            info_dict['cntr/prob'] = post_cntr_dict["prob"].item()
        info_dict["time/edit"] = edit_time

        if HAS_OUTER_DATA:
            if self.config.task == "sent":
                info_dict["edit/acc_sent"] = post_edit_dict["acc_sent"].item()

            for k, v in post_edit_dict.items():
                if isinstance(v, torch.Tensor):
                    info_dict[f"stat_dump/{k}"] = v.item()
                else:
                    info_dict[f"stat_dump/{k}"] = v

        info_dict["loss/total_supervised"] = l_total_sup.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        if not training:
            normal_val_info = super().edit_step(batch, training)[-1]
            for k, v in normal_val_info.items():
                if k not in info_dict.keys():
                    info_dict[k] = v

        return None, l_cntr, l_cls, None, info_dict
