import torch
import torch.nn as nn

from collections import OrderedDict

# Takes in latent space and predicts reward


class Reward_Predictor(nn.Module):
    # accepts input for number of transforms
    def __init__(self):
        super(Reward_Predictor, self).__init__()

        self.predictor = nn.Sequential(OrderedDict([
            ('reward_conv1', nn.Conv2d(64, 1, kernel_size=(1, 1))),
            ('reward_relu1', nn.LeakyReLU()),
            ('reward_flat1', nn.Flatten()),
            ('reward_linear1', nn.Linear(100, 1)),
            ('reward_relu2', nn.LeakyReLU()),
        ]))

    def forward(self, x):
        x = self.predictor(x)
        return x
