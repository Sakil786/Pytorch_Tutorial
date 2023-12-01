import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super.__init__()
        self.layer1 = nn.Linear(128, 32)
        self.layer2 = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, features):
        x = self.layer1(features)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


if __name__ == '__main__':
    model = Model()
    features = torch.randn((2, 128))
    model(features)
