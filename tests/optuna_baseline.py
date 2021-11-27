import argparse
import os
from typing import List
from typing import Optional
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from src.models import CompleteModel
from src.networks.extractor import ExtractorRes50
from src.networks.heads import SimpleHead
from pytorch_metric_learning import losses
from src.dataloader.wrapper import MiningDataModule
import torch
import json


def objective(trial, config):

    learning_rate = trial.suggest_float("learning_rate", 1e-7, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 0.1, log=True)
    neg_margin = trial.suggest_float("neg_margin", 0.0, 2.0)
    pos_margin = trial.suggest_float("pos_margin", 0.0, 2.0)
    datamodule = MiningDataModule(**config)
    datamodule.setup("fit")
    extractor = ExtractorRes50()
    head = SimpleHead()

    loss_func = losses.ContrastiveLoss(neg_margin=neg_margin, pos_margin=pos_margin)
    model = CompleteModel(extractor, head, loss_func, datamodule.val_set, learning_rate=learning_rate,
                          weight_decay=weight_decay)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        progress_bar_refresh_rate=500,
        val_check_interval=1.0,
        deterministic=True,
        precision=16,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="MAP@R")],
    )
    hyperparameters = dict(learning_rate=learning_rate, weight_decay=weight_decay, neg_margin=neg_margin,
                           pos_margin=pos_margin)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["MAP@R"].item()


if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('Patient Retrieval Phase1')
    parser.add_argument('--config_path', default='./config_files/', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json',
                        help='the hyper-parameter configuration and experiment settings')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)

    pruner = ( optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=config["patience"])
               if config["pruning"] else optuna.pruners.NopPruner())
    sampler = optuna.TPESampler(seed=10)
    study = optuna.create_study(direction="maximize",
                                pruner=pruner,
                                sampler=sampler,
                                study_name=config["study_name"],
                                storage=config["storage"],
                                load_if_exists=True)
    max_num_trials = 50
    num_rest_trials = max_num_trials - len(study.get_trials())

    if "prior_knowledege" in config.keys():
        study.enqueue_trial(**config["prior_knowledge"])

    if "prior_knowledege2" in config.keys():
        study.enqueue_trial(**config["prior_knowledge2"])

    study.optimize(lambda trial: objective(trial, config),
                   n_trials=num_rest_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))