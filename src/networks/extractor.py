import torch
import torch.nn as nn
from torchvision import models

class ExtractorRes50(nn.Module):
    def __init__(self, transfer_learning=True, model_path="/home/hpc/iwi5/iwi5014h/workspace/meta_files/resnet50.pth"):
        super(ExtractorRes50, self).__init__()

        self.extractor_net = models.resnet50(pretrained=False)
        if transfer_learning:
            self.extractor_net.load_state_dict(torch.load(model_path))

        self.extractor_net = nn.Sequential(*(list(self.extractor_net.children())[:-1]))

    def forward(self, x):
        x = self.extractor_net(x)
        return x

    def freeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = True

class EfficientNetB3(nn.Module):
    def __init__(self, transfer_learning=True, model_path="/home/hpc/iwi5/iwi5014h/path/to/model.pth"):
        super(EfficientNetB3, self).__init__()

        self.extractor_net = models.efficientnet_b3(pretrained=False)
        if transfer_learning:
            self.extractor_net.load_state_dict(torch.load(model_path))

        self.extractor_net = nn.Sequential(*(list(self.extractor_net.children())[:-1]))

    def forward(self, x):
        x = self.extractor_net(x)
        return x

    def freeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.extractor_net.parameters():
            param.requires_grad = True