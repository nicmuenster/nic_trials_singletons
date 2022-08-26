import sys
import os
import glob
sys.path.insert(0, '/home/hpc/iwi5/iwi5014h/nic_paper_singletons/')
from typing import List
from typing import Optional
import pytorch_lightning as pl
from src.models import CompleteModel
from src.networks.extractor import ExtractorRes50
from src.networks.heads import SimpleHead
from pytorch_metric_learning import losses
from src.dataloader.wrapper import MiningDataModule
import torch
import json
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser('Singleton Retrieval Testing Setup')
    parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
    parser.add_argument('--config', default='test_config.json',
                        help='the hyper-parameter configuration and experiment settings')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)
    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)
    # make training deterministic
    metrics_dict = {"name": [],
                "MAP@R": [],
               "r_precision": [],
               "precision_at_1": [],
               'mean_val_distance': [],
               'max_val_distance': []}
    #pl.seed_everything(42)
    for model_path in glob.glob("./**/*.pth", recursive=True):
        name = model_path.split("/")[-1].split(".")[0]
        print(model_path)
        print(name)
        metrics_dict["name"].append(name)
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()
        loss_func = losses.ContrastiveLoss()
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set)
        model.model.load_state_dict(torch.load(model_path))
        datamodule.setup("test")
        current_metrics =model.test_metrics_memory_friendly(datamodule.test_set)
        for metric_key in current_metrics.keys():
            metrics_dict[metric_key].append(current_metrics[metric_key])

    result = pd.DataFrame(metrics_dict)
    result.to_csv("./test_results_best_training.csv")

if __name__ == "__main__":
    main()