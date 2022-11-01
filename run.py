import copy
import random
import importlib
import logging

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
import utils


from trainer import EditTrainer, SupervisedTrainer
import models


OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


@hydra.main(config_path='config', config_name='config')
def run(config):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    base_dir = hydra.utils.get_original_cwd()
    LOG.info(f"Project base directory: {base_dir}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = models.get_model(config)
    tokenizer = models.get_tokenizer(config)
    if config.task == "qa" or config.task == "zsre":
        from data_classes.zsre import Seq2SeqAugmentedKILT

        if config.eval_only:
            train_set = val_set = Seq2SeqAugmentedKILT("test", tokenizer, config)
        else:
            train_set = Seq2SeqAugmentedKILT("train", tokenizer, config)
            val_set = Seq2SeqAugmentedKILT("dev", tokenizer, config)
    elif config.task == "sent":
        if "gpt" in model.name_or_path.lower():
            utils.add_padding(tokenizer, model)
        from data_classes.sentiment import SentimentDataset

        if config.eval_only:
            train_set = val_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_test.json", config)
        else:
            train_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_train.json", config)
            val_set = SentimentDataset(tokenizer, f"{base_dir}/data/sentiment/blender_val.json", config)
    elif config.task == "fnli":
        from data_classes.vitc import VitC

        if config.eval_only:
            train_set = val_set = VitC(f"{base_dir}/data/vitaminc", "test", tokenizer, config,)
        else:
            train_set = VitC(f"{base_dir}/data/vitaminc", "train", tokenizer, config)
            val_set = VitC(f"{base_dir}/data/vitaminc", "dev", tokenizer, config,)
    else:
        raise ValueError(f"Unrecognized task {config.task}")

    alg_module = importlib.import_module(f"algs.{config.alg}")
    LOG.info(f"Loading class {config.alg.upper()} from module {alg_module}")
    AlgClass = getattr(alg_module, config.alg.upper())
    alg = AlgClass(model, config, lambda: copy.deepcopy(model))

    if config.alg == "ft" and config.ft.locality.enabled:
        if config.ft.locality.oracle:
            alg.loc_sampler = train_set.edit_generator(config.ft.locality.batch_size + 1)
        else:
            state = np.random.get_state()
            np.random.seed(0)
            loc_batch = next(train_set.edit_generator(config.ft.locality.batch_size + 1))["loc"]
            np.random.set_state(state)
            alg.loc_ids = loc_batch["input_ids"]
            alg.loc_masks = loc_batch["attention_mask"]

    if config.alg == "rep" and config.rep.supervised:
        trainer = SupervisedTrainer(alg, config, train_set, val_set)
    else:
        trainer = EditTrainer(alg, config, train_set, val_set)
    LOG.info(f"Built trainer: {trainer}")
    trainer.run()


if __name__ == "__main__":
    run()
