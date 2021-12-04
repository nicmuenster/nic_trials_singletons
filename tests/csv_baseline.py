import argparse
import os
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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from pytorch_lightning.callbacks import ModelCheckpoint





if __name__ == "__main__":
    # define an argument parser
    parser = argparse.ArgumentParser('Singleton Retrieval')
    parser.add_argument('--config_path', default='./', help='the path where the config files are stored')
    parser.add_argument('--config', default='config.json',
                        help='the hyper-parameter configuration and experiment settings')
    args = parser.parse_args()
    print('Arguments:\n' + '--config_path: ' + args.config_path + '\n--config: ' + args.config)

    # read config
    with open(args.config_path + args.config, 'r') as config:
        config = config.read()

    # parse config
    config = json.loads(config)


    max_num_trials = 50
    from pathlib import Path

    hyperframe_path = config["hyper_csv"]
    if os.path.exists(config["hyper_csv"]):
        hyperframe = pd.from_csv(hyperframe_path)
    # file exists
    else:
        hyperframe = pd.DataFrame(dict(learning_rate=[],
                      weight_decay=[],
                      neg_margin=[],
                      pos_margin=[],
                      result=[]))


    if len(hyperframe["result"]) == 0:

        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()

        loss_func = losses.ContrastiveLoss(neg_margin=config["initial_trial1"]["neg_margin"], pos_margin=config["initial_trial1"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set, learning_rate=10^config["initial_trial1"]["learning_rate"],
                              weight_decay=10^config["initial_trial1"]["weight_decay"])
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_folder"],
            filename=config["checkpoint_name"])
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            resume_from_checkpoint=checkpoint,
            callbacks=[checkpoint_callback],
            progress_bar_refresh_rate=500,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        hyperparameters = dict(learning_rate=10 ^ config["initial_trial1"]["learning_rate"],
                               weight_decay=10 ^ config["initial_trial1"]["weight_decay"],
                               neg_margin=config["initial_trial1"]["neg_margin"],
                               pos_margin=config["initial_trial1"]["pos_margin"])
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        result = dict(learning_rate=[10 ^ config["initial_trial1"]["learning_rate"]],
               weight_decay=[10 ^ config["initial_trial1"]["weight_decay"]],
               neg_margin=[config["initial_trial1"]["neg_margin"]],
               pos_margin=[config["initial_trial1"]["pos_margin"]],
                        result=[trainer.callback_metrics["MAP@R"].item()])

        torch.save(model.model.state_dict(),
                   config["best_model_path"])
        hyperframe  = pd.DataFrame(result)
        hyperframe.to_csv(hyperframe_path)
        if os.path.exists(config["checkpoint_path"]):
            os.remove(config["checkpoint_path"])






    if len(hyperframe["result"]) == 1:
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()

        loss_func = losses.ContrastiveLoss(neg_margin=config["initial_trial2"]["neg_margin"], pos_margin=config["initial_trial2"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ^ config["initial_trial2"]["learning_rate"],
                              weight_decay=10 ^ config["initial_trial2"]["weight_decay"])
        # TODO fill in clause for case of resumed training
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_folder"],
            filename=config["checkpoint_name"])
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            resume_from_checkpoint=checkpoint,
            callbacks=[checkpoint_callback],
            progress_bar_refresh_rate=500,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        hyperparameters = dict(learning_rate=config["initial_trial2"]["learning_rate"],
                               weight_decay=config["initial_trial2"]["weight_decay"],
                               neg_margin=config["initial_trial2"]["neg_margin"],
                               pos_margin=config["initial_trial2"]["pos_margin"])
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        result = dict(learning_rate=config["initial_trial2"]["learning_rate"],
                      weight_decay=config["initial_trial2"]["weight_decay"],
                      neg_margin=config["initial_trial2"]["neg_margin"],
                      pos_margin=config["initial_trial2"]["pos_margin"],
                      result=trainer.callback_metrics["MAP@R"].item())
        if (hyperframe["result"].to_list()[-1] <= result["result"]):
            torch.save(model.model.state_dict(),
                       config["best_model_path"])
        result = pd.DataFrame(result)

        frame_list = [hyperframe, result]

        hyperframe = pd.concat(frame_list)
        hyperframe.to_csv(hyperframe_path)
        if os.path.exists(config["checkpoint_path"]):
            os.remove(config["checkpoint_path"])


    bayes_opt = BayesianOptimization(
        f=None,
        pbounds=config["param_bounds"],
        verbose=2,
        random_state=1,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    dict_list = hyperframe.to_dict('records')
    for entry in dict_list:
        params = {"learning_rate": entry["learning_rate"],
                  "weight_decay": entry["weight_decay"],
                  "neg_margin": entry["neg_margin"],
                  "pos_margin": entry["pos_margin"], }
        bayes_opt.register(params=params, target=entry["result"])

    if not (os.path.exists(config["checkpoint_path"]) or config["checkpoint"]):
        new_hyperparams = bayes_opt.suggest(utility)
        config["intermediate_save"] = new_hyperparams
        config["checkpoint"] = True
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)





    for x in range(config["num_iter"] - len(hyperframe)):

        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        loss_func = losses.ContrastiveLoss(neg_margin=config["intermediate_save"]["neg_margin"], pos_margin=config["intermediate_save"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ^ config["intermediate_save"]["learning_rate"],
                              weight_decay=10 ^ config["intermediate_save"]["weight_decay"])
        # TODO fill in clause for case of resumed training
        checkpoint_callback = ModelCheckpoint(
        dirpath = config["checkpoint_folder"],
        filename = config["checkpoint_name"])
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            resume_from_checkpoint=checkpoint,
            callbacks=[checkpoint_callback],
            progress_bar_refresh_rate=500,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        hyperparameters = dict(learning_rate=new_hyperparams["learning_rate"],
                               weight_decay=new_hyperparams["weight_decay"],
                               neg_margin=new_hyperparams["neg_margin"],
                               pos_margin=new_hyperparams["pos_margin"])
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule)

        result = dict(learning_rate=new_hyperparams["learning_rate"],
                      weight_decay=new_hyperparams["weight_decay"],
                      neg_margin=new_hyperparams["neg_margin"],
                      pos_margin=new_hyperparams["pos_margin"],
                      result=trainer.callback_metrics["MAP@R"].item())
        resultdf = pd.DataFrame(result)
        frame_list = [hyperframe, resultdf]
        hyperframe = pd.concat(frame_list)
        hyperframe.to_csv(hyperframe_path)
        if (bayes_opt.max["target"] <= result["result"]):
            torch.save(model.model.state_dict(),
                       config["best_model_path"])

        params = {"learning_rate": result["learning_rate"],
                  "weight_decay": result["weight_decay"],
                  "neg_margin": result["neg_margin"],
                  "pos_margin": result["pos_margin"], }
        config["checkpoint"] = False
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)
        os.remove(config["checkpoint_path"])
        bayes_opt.register(params=params, target=entry["result"])
        new_hyperparams = bayes_opt.suggest(utility)
        config["intermediate_save"] = new_hyperparams
        config["checkpoint"] = True
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)

    config["finished_optimization"] = True
    with open(args.config_path + args.config, "w") as config_out:
        json.dump(config, config_out)






