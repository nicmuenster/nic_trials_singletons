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
    # to make code backward compatible to earlier versions without lr_scheduler
    if "lr_scheduler" not in config.keys():
        config["lr_scheduler"] = False
    # for the moment, the amount of optimization steps is initialized here. Not pretty but works fine
    # for the moment and easier than current alternatives. Candidate to be changed later on.
    max_num_trials = 50
    # config["num_iter"] = max_num_trials

    # so if not new_hyperparams works
    new_hyperparams = []
    hyperframe_path = config["hyper_csv"]
    # file exists
    if os.path.exists(config["hyper_csv"]):
        hyperframe = pd.read_csv(hyperframe_path, names=['learning_rate',
                                                         'weight_decay',
                                                         'neg_margin',
                                                         'pos_margin',
                                                         'req_epochs',
                                                         'result'])
    # else create csv file for bayesian optimization
    else:
        hyperframe = pd.DataFrame(dict(learning_rate=[],
                                       weight_decay=[],
                                       neg_margin=[],
                                       pos_margin=[],
                                       req_epochs=[],
                                       result=[]))
    # rename last checkpoint if necessary
    condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
    # first iteration with given hyperparams
    if len(hyperframe["result"]) == 0:
        # init data module and inner network
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()
        # innit loss and wrapper network
        loss_func = losses.ContrastiveLoss(neg_margin=config["initial_trial1"]["neg_margin"],
                                           pos_margin=config["initial_trial1"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ** config["initial_trial1"]["learning_rate"],
                              weight_decay=10 ** config["initial_trial1"]["weight_decay"],
                              lr_scheduler=config["lr_scheduler"])
        # check for existing checkpoint
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        # init callbacks and trainer
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_folder"],
            filename=config["checkpoint_name"])
        early_stop_callback = EarlyStopping(monitor="MAP@R", min_delta=0.005, patience=config["patience"],
                                            verbose=False,
                                            mode="max",
                                            divergence_threshold=0.05)
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            max_epochs=50,
            min_epochs=15,
            callbacks=[checkpoint_callback, early_stop_callback],
            #progress_bar_refresh_rate=500,
            enable_progress_bar=False,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        # to safe hyperparams in the log files as well
        hyperparameters = dict(learning_rate=config["initial_trial1"]["learning_rate"],
                               weight_decay=config["initial_trial1"]["weight_decay"],
                               neg_margin=config["initial_trial1"]["neg_margin"],
                               pos_margin=config["initial_trial1"]["pos_margin"])
        trainer.logger.log_hyperparams(hyperparameters)
        # fit model
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)

        result = dict(learning_rate=[config["initial_trial1"]["learning_rate"]],
                      weight_decay=[config["initial_trial1"]["weight_decay"]],
                      neg_margin=[config["initial_trial1"]["neg_margin"]],
                      pos_margin=[config["initial_trial1"]["pos_margin"]],
                      req_epochs=[model.epoch_counter],
                      result=[trainer.callback_metrics["MAP@R"].item()])
        # save current model
        torch.save(model.model.state_dict(),
                   config["best_model_path"])
        # save hyperparams and corresponding result to csv
        hyperframe = pd.DataFrame(result)
        hyperframe.to_csv(hyperframe_path)
        # remove current checkpoint, since iteration finished correctly
        # once this point within the code is reached
        condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
        if os.path.exists(config["checkpoint_path"]):
            os.remove(config["checkpoint_path"])

    # second iteration with given hyperparams, almost the same as the first
    if len(hyperframe["result"]) == 1:
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()

        loss_func = losses.ContrastiveLoss(neg_margin=config["initial_trial2"]["neg_margin"],
                                           pos_margin=config["initial_trial2"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ** config["initial_trial2"]["learning_rate"],
                              weight_decay=10 ** config["initial_trial2"]["weight_decay"],
                              lr_scheduler=config["lr_scheduler"])
        condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_folder"],
            filename=config["checkpoint_name"])
        early_stop_callback = EarlyStopping(monitor="MAP@R", min_delta=0.005,
                                            patience=config["patience"], verbose=False,
                                            mode="max",
                                            divergence_threshold=0.05)
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            max_epochs=50,
            min_epochs=20,
            callbacks=[checkpoint_callback, early_stop_callback],
            #progress_bar_refresh_rate=500,
            enable_progress_bar=False,
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
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)

        result = dict(learning_rate=[config["initial_trial2"]["learning_rate"]],
                      weight_decay=[config["initial_trial2"]["weight_decay"]],
                      neg_margin=[config["initial_trial2"]["neg_margin"]],
                      pos_margin=[config["initial_trial2"]["pos_margin"]],
                      req_epochs=[model.epoch_counter],
                      result=[trainer.callback_metrics["MAP@R"].item()])
        # update best model if necessary
        if hyperframe["result"].to_list()[-1] <= result["result"][-1]:
            torch.save(model.model.state_dict(),
                       config["best_model_path"])
        result = pd.DataFrame(result)

        frame_list = [hyperframe, result]

        hyperframe = pd.concat(frame_list, ignore_index=True)
        hyperframe.to_csv(hyperframe_path)
        condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])

        if os.path.exists(config["checkpoint_path"]):
            os.remove(config["checkpoint_path"])

    # beginning of actual bayesian optimization
    # init optimizer
    bayes_opt = BayesianOptimization(
        f=None,
        pbounds=config["param_bounds"],
        verbose=2,
        random_state=1,
    )
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    # feed all current hyperparam entries to the optimizer
    dict_list = hyperframe.to_dict('records')
    for entry in dict_list:
        params = {"learning_rate": entry["learning_rate"],
                  "weight_decay": entry["weight_decay"],
                  "neg_margin": entry["neg_margin"],
                  "pos_margin": entry["pos_margin"], }
        bayes_opt.register(params=params, target=entry["result"])
    # once again condense checkpoints
    condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
    # if there is no current checkpoint, get new hyperparams and save them in the config
    if not (os.path.exists(config["checkpoint_path"]) or config["checkpoint"]):
        new_hyperparams = bayes_opt.suggest(utility)
        config["intermediate_save"] = new_hyperparams
        config["checkpoint"] = True
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)

    # loop for the rest of the script, initialzed with the remaining amount of predefined steps
    for x in range(config["num_iter"] - len(hyperframe)):
        # identical to earlier part
        datamodule = MiningDataModule(**config)
        datamodule.setup("fit")
        extractor = ExtractorRes50()
        head = SimpleHead()
        condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
        checkpoint = config["checkpoint_path"] if os.path.exists(config["checkpoint_path"]) else None
        loss_func = losses.ContrastiveLoss(neg_margin=config["intermediate_save"]["neg_margin"],
                                           pos_margin=config["intermediate_save"]["pos_margin"])
        model = CompleteModel(extractor, head, loss_func, datamodule.val_set,
                              learning_rate=10 ** config["intermediate_save"]["learning_rate"],
                              weight_decay=10 ** config["intermediate_save"]["weight_decay"],
                              lr_scheduler=config["lr_scheduler"])
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["checkpoint_folder"],
            filename=config["checkpoint_name"])
        early_stop_callback = EarlyStopping(monitor="MAP@R", min_delta=0.005,
                                            patience=config["patience"], verbose=False,
                                            mode="max",
                                            divergence_threshold=0.05)
        trainer = pl.Trainer(
            logger=True,
            default_root_dir=config["log_path"],
            max_epochs=50,
            min_epochs=20,
            callbacks=[checkpoint_callback, early_stop_callback],
            #progress_bar_refresh_rate=500,
            enable_progress_bar=False,
            val_check_interval=1.0,
            deterministic=True,
            precision=16,
            gpus=1 if torch.cuda.is_available() else None,
        )
        # load hyperparams, that are either newly created before the loop or at
        # the end of its iteration, or from immediate checkpoint if there is one
        if not new_hyperparams:
            new_hyperparams = config["intermediate_save"]
        hyperparameters = dict(learning_rate=new_hyperparams["learning_rate"],
                               weight_decay=new_hyperparams["weight_decay"],
                               neg_margin=new_hyperparams["neg_margin"],
                               pos_margin=new_hyperparams["pos_margin"])
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint)
        # log and save results
        result = dict(learning_rate=[new_hyperparams["learning_rate"]],
                      weight_decay=[new_hyperparams["weight_decay"]],
                      neg_margin=[new_hyperparams["neg_margin"]],
                      pos_margin=[new_hyperparams["pos_margin"]],
                      req_epochs=[model.epoch_counter],
                      result=[trainer.callback_metrics["MAP@R"].item()])
        resultdf = pd.DataFrame(result)
        frame_list = [hyperframe, resultdf]
        hyperframe = pd.concat(frame_list, ignore_index=True)
        hyperframe.to_csv(hyperframe_path)
        # save model if the current one was the best, update config
        if (bayes_opt.max["target"] <= result["result"][-1]):
            torch.save(model.model.state_dict(),
                       config["best_model_path"])
        params = {"learning_rate": result["learning_rate"][0],
                  "weight_decay": result["weight_decay"][0],
                  "neg_margin": result["neg_margin"][0],
                  "pos_margin": result["pos_margin"][0], }
        config["checkpoint"] = False
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)
        # again, remove checkpoint if one exists
        condense_checkpoints(config["checkpoint_folder"], config["checkpoint_name"])
        if os.path.exists(config["checkpoint_path"]):
            os.remove(config["checkpoint_path"])
        # new bayes step
        bayes_opt.register(params=params, target=entry["result"])
        new_hyperparams = bayes_opt.suggest(utility)
        config["intermediate_save"] = new_hyperparams
        config["checkpoint"] = True
        with open(args.config_path + args.config, "w") as config_out:
            json.dump(config, config_out)

    config["finished_optimization"] = True
    with open(args.config_path + args.config, "w") as config_out:
        json.dump(config, config_out)
