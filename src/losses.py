import torch.nn as nn
from pytorch_metric_learning import losses


class LossWrapper(nn.Module):

    def __init__(self, loss_func, embedding_size=128, miner=None, cbm_size=None):
        super(LossWrapper, self).__init__()
        self.embedding_size = embedding_size
        self.cbm_size = cbm_size
        self.miner = miner
        self.loss_func = loss_func
        if self.cbm_size is not None:
            self.loss_func = losses.CrossBatchMemory(self.loss_func, embedding_size, memory_size=cbm_size,
                                                     miner=self.miner)

    def forward(self, embeddings, labels):
        if self.miner:
            indices = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, indices)
        else:
            loss = self.loss_func(embeddings, labels)
        return loss