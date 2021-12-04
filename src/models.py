import pytorch_lightning as pl
import numpy as np
import faiss
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
from pytorch_metric_learning import miners, losses, reducers, distances, regularizers
import torch.nn as nn
import torch
import time


class CompleteModel(pl.LightningModule):
    def __init__(self, extractor, head, loss_func, test_set_val, learning_rate=1e-5, weight_decay=1e-6):
        super().__init__()
        self.save_hyperparameters( 'learning_rate',
                                   'weight_decay')
        self.accuracy_calculator = AccuracyCalculator(
            include=("mean_average_precision_at_r", "precision_at_1", "r_precision"), avg_of_avgs=False)
        self.loss_func = loss_func
        # used for val logging
        self.test_set_val = test_set_val

        self.model = nn.Sequential(extractor, head)
        self.distance = distances.LpDistance(normalize_embeddings=False)

    def forward(self, x):
        x = self.model(x)
        return x

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {'params': params, 'weight_decay': weight_decay},
            {'params': excluded_params, 'weight_decay': 0.},
        ]

    def configure_optimizers(self):
        params_to_train = self.exclude_from_wt_decay(self.model.named_parameters(),
                                                         weight_decay=self.hparams.weight_decay)
        optimizer = torch.optim.AdamW(params_to_train,
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay)


        return [optimizer],

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        dist_mat = self.distance.compute_mat(embeddings.type(torch.float), None)
        mean_distance = torch.mean(dist_mat)
        max_distance = torch.max(dist_mat)
        metrics = {
            'train_loss': loss, 'mean_distance': mean_distance, 'max_distance': max_distance}
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        # since val loss is not really a metric we look at, we simply skip computing it
        return None

    def validation_epoch_end(self, outs):
        self.test_various_metrics(self.test_set_val)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        embeddings = self(inputs)
        loss = self.loss_func(embeddings, labels)
        self.log('test_loss', loss)

    # most likely useless as model has to be created again (at least when using lr finder)
    def unfreeze_extractor(self):
        self.hparams.extractor.unfreeze()
        self.hparams.frozen_at_start = False

    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(dataloader_num_workers=4)
        return tester.get_all_embeddings(dataset, model)

    def test_various_metrics(self, testset):
        t0 = time.time()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        embeddings, labels = self.get_all_embeddings(testset, self.model)
        print("Computing accuracy")
        accuracies = self.accuracy_calculator.get_accuracy(embeddings,
                                                           embeddings,
                                                           np.squeeze(labels),
                                                           np.squeeze(labels),
                                                           True)
        dist_mat = self.distance.compute_mat(torch.tensor(embeddings, dtype=torch.float), None)
        mean_distance = torch.mean(dist_mat)
        max_distance = torch.max(dist_mat)
        print("mean_distance = " + str(mean_distance))
        print("max_distance = " + str(max_distance))
        print("Test set accuracy (MAP@R) = {}".format(accuracies["mean_average_precision_at_r"]))
        print("r_prec = " + str(accuracies["r_precision"]))
        print("prec_at_1 = " + str(accuracies["precision_at_1"]))
        t1 = time.time()
        print("Time used for evaluating: " + str((t1 - t0) / 60) + " minutes")
        metrics = {"MAP@R": accuracies["mean_average_precision_at_r"],
                   "r_precision": accuracies["r_precision"],
                   "precision_at_1": accuracies["precision_at_1"],
                   'mean_val_distance': mean_distance,
                   'max_val_distance': max_distance
                   }
        self.log_dict(metrics)
