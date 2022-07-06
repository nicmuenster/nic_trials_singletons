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
    def __init__(self, extractor, head, loss_func, test_set_val, learning_rate=1e-5, weight_decay=1e-6,
                 lr_scheduler=False):
        super().__init__()
        self.save_hyperparameters('learning_rate',
                                   'weight_decay', 'lr_scheduler', ignore=['extractor', 'head', 'loss_func'])
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

        if self.hparams.lr_scheduler:
            # standard pytorch params except "min" to "max" and patience from 10 to 5
            scheduler = torch.optim.lr_scheduler.torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                                                            factor=0.1,
                                                                                            patience=5,
                                                                                            threshold=0.0001,
                                                                                            threshold_mode='rel',
                                                                                            cooldown=0, min_lr=0,
                                                                                            eps=1e-08,
                                                                                            verbose=False)
            # scheduler config used by lightning, for clarity descriptions were ported as well
            lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "epoch",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "MAP@R",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": None,
            }

            return [optimizer], [lr_scheduler_config]


        return [optimizer]

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

    def get_all_embeddings(self, dataset, model):
        tester = testers.BaseTester(dataloader_num_workers=4)
        return tester.get_all_embeddings(dataset, model)

    def test_various_metrics_older_version(self, testset):
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
        return metrics


    def test_various_metrics(self, testset):
        t0 = time.time()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        embeddings, labels = self.get_all_embeddings(testset, self.model)
        print(embeddings.size())
        print(labels.size())
        labels = labels.squeeze(1)
        print(labels.size())
        print("Computing accuracy")
        accuracies = self.accuracy_calculator.get_accuracy(embeddings,
                                                           embeddings,
                                                           labels,
                                                           labels,
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
        return metrics


    def test_metrics_memory_friendly(self, testset):
        t0 = time.time()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        embeddings, labels = self.get_all_embeddings(testset, self.model)
        print("Computing accuracy")
        sample_counter = 0
        accuracies = {"mean_average_precision_at_r": 0,
                      "precision_at_1": 0,
                      "r_precision": 0,
                      }
        max_distance = 0
        mean_distance = 0
        class_list, class_count = np.unique(labels.clone().cpu(), return_counts=True)
        for class_instance, class_number in zip(class_list, class_count):
            if class_number > 1:
                sample_counter = sample_counter + class_number
                current_indices = np.argwhere(labels.clone().cpu() == class_instance)[0]
                embedding_subset = embeddings[current_indices]
                label_subset = labels[current_indices]
                #print(embedding_subset.size())
                #print(embeddings.size())
                #print(label_subset.size())
                #print(labels.size())
                label_subset = label_subset.squeeze(-1)
                labels = labels.squeeze(-1)
                #print(label_subset.size())
                #print(labels.size())

                intermediate_accuracies = self.accuracy_calculator.get_accuracy(embedding_subset,
                                                                                embeddings,
                                                                                label_subset,
                                                                                labels,
                                                                                True)
                accuracies["mean_average_precision_at_r"] = accuracies["mean_average_precision_at_r"] + \
                                                            intermediate_accuracies["mean_average_precision_at_r"] \
                                                            * class_number
                accuracies["r_precision"] = accuracies["r_precision"] + \
                                            intermediate_accuracies["r_precision"] * class_number
                accuracies["precision_at_1"] = accuracies["precision_at_1"] + \
                                               intermediate_accuracies["precision_at_1"] * class_number
                int_dist_mat = self.distance.compute_mat(torch.tensor(embedding_subset, dtype=torch.float), None)
                int_mean_distance = torch.mean(int_dist_mat) * class_number
                mean_distance = mean_distance + int_mean_distance
                int_max_distance = torch.max(int_dist_mat)
                if int_max_distance > max_distance:
                    max_distance = int_max_distance
        mean_distance = mean_distance / sample_counter
        accuracies["mean_average_precision_at_r"] = accuracies["mean_average_precision_at_r"] / sample_counter
        accuracies["r_precision"] = accuracies["r_precision"] / sample_counter
        accuracies["precision_at_1"] = accuracies["precision_at_1"] / sample_counter
        print("mean_interclass_distance = " + str(mean_distance))
        print("max_interclass_distance = " + str(max_distance))
        print("Test set accuracy (MAP@R) = {}".format(accuracies["mean_average_precision_at_r"]))
        print("r_prec = " + str(accuracies["r_precision"]))
        print("prec_at_1 = "  + str(accuracies["precision_at_1"]))
        t1 = time.time()
        print("Time used for evaluating: " + str((t1 - t0) / 60) + " minutes")
        metrics = {"MAP@R": accuracies["mean_average_precision_at_r"],
                   "r_precision": accuracies["r_precision"],
                   "precision_at_1": accuracies["precision_at_1"],
                   'mean_val_distance': mean_distance.cpu().item(),
                   'max_val_distance': max_distance.cpu().item()
                   }
        #self.log_dict(metrics)
        return metrics


    def test_metrics_memory_friendly_old_version(self, testset):
        t0 = time.time()
        device = torch.device("cuda")
        self.model = self.model.to(device)
        embeddings, labels = self.get_all_embeddings(testset, self.model)
        print("Computing accuracy")
        sample_counter = 0
        accuracies = {"mean_average_precision_at_r": 0,
                      "precision_at_1": 0,
                      "r_precision": 0,
                      }
        max_distance = 0
        mean_distance = 0
        class_list, class_count = np.unique(labels.clone().cpu(), return_counts=True)
        for class_instance, class_number in zip(class_list, class_count):
            if class_number > 1:
                sample_counter = sample_counter + class_number
                current_indices = np.argwhere(labels.clone().cpu() == class_instance)[0]
                embedding_subset = embeddings[current_indices]
                label_subset = labels[current_indices]
                intermediate_accuracies = self.accuracy_calculator.get_accuracy(embedding_subset,
                                                                                embeddings,
                                                                                np.squeeze(label_subset),
                                                                                np.squeeze(labels),
                                                                                True)
                accuracies["mean_average_precision_at_r"] = accuracies["mean_average_precision_at_r"] + \
                                                            intermediate_accuracies["mean_average_precision_at_r"] \
                                                            * class_number
                accuracies["r_precision"] = accuracies["r_precision"] + \
                                            intermediate_accuracies["r_precision"] * class_number
                accuracies["precision_at_1"] = accuracies["precision_at_1"] + \
                                               intermediate_accuracies["precision_at_1"] * class_number
                int_dist_mat = self.distance.compute_mat(torch.tensor(embedding_subset, dtype=torch.float), None)
                int_mean_distance = torch.mean(int_dist_mat) * class_number
                mean_distance = mean_distance + int_mean_distance
                int_max_distance = torch.max(int_dist_mat)
                if int_max_distance > max_distance:
                    max_distance = int_max_distance
        mean_distance = mean_distance / sample_counter
        accuracies["mean_average_precision_at_r"] = accuracies["mean_average_precision_at_r"] / sample_counter
        accuracies["r_precision"] = accuracies["r_precision"] / sample_counter
        accuracies["precision_at_1"] = accuracies["precision_at_1"] / sample_counter
        print("mean_interclass_distance = " + str(mean_distance))
        print("max_interclass_distance = " + str(max_distance))
        print("Test set accuracy (MAP@R) = {}".format(accuracies["mean_average_precision_at_r"]))
        print("r_prec = " + str(accuracies["r_precision"]))
        print("prec_at_1 = "  + str(accuracies["precision_at_1"]))
        t1 = time.time()
        print("Time used for evaluating: " + str((t1 - t0) / 60) + " minutes")
        metrics = {"MAP@R": accuracies["mean_average_precision_at_r"],
                   "r_precision": accuracies["r_precision"],
                   "precision_at_1": accuracies["precision_at_1"],
                   'mean_val_distance': mean_distance.cpu().item(),
                   'max_val_distance': max_distance.cpu().item()
                   }
        #self.log_dict(metrics)
        return metrics

