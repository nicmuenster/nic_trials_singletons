import torch.nn as nn

class SimpleHead(nn.Module):
    def __init__(self, input_size=2048, embedding_size=128):
        super(SimpleHead, self).__init__()
        self.fc = nn.Sequential(nn.Flatten(),  nn.Linear(input_size, embedding_size))

    def forward(self, x):
        return self.fc(x)
