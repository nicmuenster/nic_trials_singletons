import argparse
import os
import sys

sys.path.insert(0, '/home/hpc/iwi5/iwi5014h/nic_paper_singletons/')
from typing import List
from typing import Optional
import pytorch_lightning as pl
from src.models import CompleteModel
from src.networks.extractor import ExtractorRes50
from src.networks.heads import SimpleHead
from pytorch_metric_learning import losses
from src.dataloader.wrapper import MiningDataModule
from src.utils.utils import condense_checkpoints
import torch
import json
import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == "__main__":
    # make training deterministic
    pl.seed_everything(42)

    parser = argparse.ArgumentParser('Singleton Retrieval Testing Setup')
    parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json',
                        help='the general experiment settings')
    parser.add_argument('--model_params', default='model_params.csv',
                        help='the path to the hyper-parameter configurations for the models')
    parser.add_argument('--fold', default='1',
                        help='which fold shoud be used')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)
    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)

    model_params = pd.read_csv(args.config_path + args.model_params)
    hyperframe_path = "./fold" + args.fold + "_results.csv"
    if os.path.exists(hyperframe_path):
        hyperframe = pd.read_csv(hyperframe_path)
    else:
        hyperframe = pd.DataFrame({"learning_rate":[],
                                       "weight_decay":[],
                                       "neg_margin":[],
                                       "pos_margin":[],
                                       "result":[],
                                       "MAP@R_test":[],
                                       "r_precision":[],
                                       "precision_at_1_test":[],
                                       "mean_val_distance_test":[],
                                       "max_val_distance_test":[]})
    if config["csv_test"].contains(".csv"):
        config["csv_val"] = config["csv_val"].split("val")[0]
        config["csv_train"] = config["csv_train"].split("train")[0]
        config["csv_val_singles"] = config["csv_val"]
        config["csv_train_singles"] = config["csv_train"]
    config["csv_val"] = config["csv_val"] + "val" + args.fold + ".csv"
    config["csv_train"] = config["csv_train"] + "train" + args.fold + ".csv"
    config["csv_val_singles"] = config["csv_val_singles"] + "val_singles" + args.fold + ".csv"
    config["csv_train_singles"] = config["csv_train_singles"] + "train_singles" + args.fold + ".csv"


    for row in model_params.itertuples(index=True, name='Pandas'):
        if row.name.contains("std"):
            continue
        # init data module and inner network
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()
        # innit loss and wrapper network
        loss_func = losses.ContrastiveLoss(neg_margin=row.neg_margin,
                                           pos_margin=row.pos_margin)
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ** row.learning_rate,
                              weight_decay=10 ** row.weight_decay)
        early_stop_callback = EarlyStopping(monitor="MAP@R", min_delta=0.005, patience=config["patience"],
                                            verbose=False,
                                            mode="max",
                                            divergence_threshold=0.05)
        trainer = pl.Trainer(
            logger=True,
            max_epochs=50,
            min_epochs=20,
            callbacks=[early_stop_callback],
            progress_bar_refresh_rate=500,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        # to safe hyperparams in the log files as well
        hyperparameters = dict(learning_rate=row.learning_rate,
                              weight_decay=row.weight_decay,
                               neg_margin=row.neg_margin,
                               pos_margin=row.pos_margin)
        trainer.logger.log_hyperparams(hyperparameters)
        # fit model
        trainer.fit(model, datamodule=datamodule)
        torch.save(model.model.state_dict(),
                   row.name + ".pth")
        result = dict(learning_rate=[row.learning_rate],
                              weight_decay=[row.weight_decay],
                               neg_margin=[row.neg_margin],
                               pos_margin=[row.pos_margin],
                      result=[trainer.callback_metrics["MAP@R"].item()])
        # save current model
        torch.save(model.model.state_dict(),
                   args.config_path + row.name + ".pth")
        datamodule.setup("test")
        current_metrics = model.test_metrics_memory_friendly(datamodule.test_set)
        for metric_key in current_metrics.keys():
            result[metric_key + "_test"] = [current_metrics[metric_key]]
        # save hyperparams and corresponding result to csv
        frame_list = [hyperframe, result]
        hyperframe = pd.concat(frame_list)
        hyperframe.to_csv(hyperframe_path)
