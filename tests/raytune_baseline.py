from ray import tune
from ray_lightning import RayPlugin
from ray_lightning.tune import TuneReportCallback, get_tune_ddp_resources
import json
import pytorch_lightning as pl
from src.models import CompleteModel
from src.networks.extractor import ExtractorRes50
from src.networks.heads import SimpleHead
from pytorch_metric_learning import losses
from src.dataloader.wrapper import MiningDataModule
import torch
import argparse
import math
import os
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback


config = {
    "layer_1": tune.choice([32, 64, 128]),
    "layer_2": tune.choice([64, 128, 256]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
}

def train_mnist_tune(config, num_epochs=10, num_gpus=0, data_dir="~/data"):
    learning_rate = tune.loguniform(config["learning_rate"][0], config["learning_rate"][1])
    weight_decay = tune.loguniform(config["weight_decay"][0], config["weight_decay"][1])
    neg_margin = tune.loguniform(config["neg_margin"][0], config["neg_margin"][1])
    pos_margin = tune.loguniform(config["pos_margin"][0], config["pos_margin"][1])
    datamodule = MiningDataModule(**config)
    datamodule.setup("fit")
    extractor = ExtractorRes50()
    head = SimpleHead()

    loss_func = losses.ContrastiveLoss(neg_margin=neg_margin, pos_margin=pos_margin)
    model = CompleteModel(extractor, head, loss_func, datamodule.val_set, learning_rate=learning_rate,
                          weight_decay=weight_decay)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                on="validation_end")
        ])
    trainer.fit(model)


def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
    config = {
        "layer_1_size": tune.choice([32, 64, 128]),
        "layer_2_size": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_mnist_tune,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            data_dir=data_dir),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_mnist_asha")

    print("Best hyperparameters found were: ", analysis.best_config)


callback = TuneReportCheckpointCallback(
    metrics={"loss": "val_loss", "mean_accuracy": "val_accuracy"},
    filename="checkpoint",
    on="validation_end")

def train_mnist_tune_checkpoint(config,
                                checkpoint_dir=None,
                                num_epochs=10,
                                num_gpus=0,
                                data_dir="~/data"):
    data_dir = os.path.expanduser(data_dir)
    kwargs = {
        "max_epochs": num_epochs,
        # If fractional GPUs passed in, convert to int.
        "gpus": math.ceil(num_gpus),
        "logger": TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        "progress_bar_refresh_rate": 0,
        "callbacks": [
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                filename="checkpoint",
                on="validation_end")
        ]
    }

    if checkpoint_dir:
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpoint")

    model = LightningMNISTClassifier(config=config, data_dir=data_dir)
    trainer = pl.Trainer(**kwargs)

    trainer.fit(model)

def tune_mnist_pbt(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
        config = {
            "layer_1_size": tune.choice([32, 64, 128]),
            "layer_2_size": tune.choice([64, 128, 256]),
            "lr": 1e-3,
            "batch_size": 64,
        }

        scheduler = PopulationBasedTraining(
            perturbation_interval=4,
            hyperparam_mutations={
                "lr": tune.loguniform(1e-4, 1e-1),
                "batch_size": [32, 64, 128]
            })

        reporter = CLIReporter(
            parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
            metric_columns=["loss", "mean_accuracy", "training_iteration"])

        analysis = tune.run(
            tune.with_parameters(
                train_mnist_tune_checkpoint,
                num_epochs=num_epochs,
                num_gpus=gpus_per_trial,
                data_dir=data_dir),
            resources_per_trial={
                "cpu": 1,
                "gpu": gpus_per_trial
            },
            metric="loss",
            mode="min",
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="tune_mnist_pbt")

        print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('Raytune Baseline')
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

    # Create plugin.
    ray_plugin = RayPlugin(num_workers=4, use_gpu=True)

    # Report loss and accuracy to Tune after each validation epoch:
    trainer = pl.Trainer(plugins=[ray_plugin], callbacks=[
        TuneReportCallback(["val_loss", "val_acc"],
                           on="validation_end")])

    # Same as above, but report as `loss` and `mean_accuracy`:
    trainer = pl.Trainer(plugins=[ray_plugin], callbacks=[
        TuneReportCallback(
            {"loss": "val_loss", "mean_accuracy": "val_acc"},
            on="validation_end")])


