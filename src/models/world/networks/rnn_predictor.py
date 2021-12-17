import torch
import torch.nn as nn

from collections import OrderedDict

from .conv_lstm import ConvLSTM
from .general import LambdaLayer

'''
Transform_Autoencoder Class

Train an autoencoder that accepts additional image transformative inputs
'''


class RNNPredictor(nn.Module):
    # accepts input for number of transforms
    def __init__(self, num_transforms=0, frame_stacks=1):
        super(RNNPredictor, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(3 * frame_stacks, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.ReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.ReLU()),
            ('encoder_conv3', nn.Conv2d(32, 64, kernel_size=7)),
            ('encoder_relu3', nn.LeakyReLU()),
        ]))

        self.bottleneck = nn.Sequential(OrderedDict([
            ('bottleneck_conv1', nn.Conv2d(64+num_transforms, 64, kernel_size=(1, 1))),
            ('bottleneck_relu1', nn.ReLU()),
        ]))

        self.lstm = nn.Sequential(OrderedDict([
            ('conv_lstm1', ConvLSTM(64, [64, 64, 128, 64], kernel_size=(1, 1), num_layers=4))
        ]))

        self.reward_predictor = nn.Sequential(OrderedDict([
            ('reward_conv1', nn.Conv2d(64, 16, kernel_size=7)),
            ('reward_relu1', nn.ReLU()),
            ('reward_flat1', nn.Flatten()),
            ('reward_linear1', nn.Linear(256, 64)),
            ('reward_relu2', nn.ReLU()),
            ('reward_linear2', nn.Linear(64, 1)),
            ('reward_sig1', nn.Sigmoid()),
            ('reward_clamp1', LambdaLayer(lambda x: x * 240 - 100))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv1', nn.ConvTranspose2d(64, 32, kernel_size=7)),
            ('decoder_relu1', nn.ReLU()),
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu2', nn.ReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, 3 * frame_stacks, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu3', nn.LeakyReLU()),
        ]))

    # forward pass takes initial image and array of transforms
    def forward(self, x, transforms=[]):
        # print(x.shape)
        x = self.encoder(x)
        # print(x.shape)
        # print(transforms.shape)
        x = torch.cat((x, transforms), 1)
        # print(x.shape)
        x = self.bottleneck(x)
        # t = torch.tensor([i for i in range(0, 128)]).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to("cuda")

        # xt = torch.cat((x.unsqueeze(0), t), dim=0).to("cuda")
        # print(xt.shape)
        x = x.unsqueeze(0)
        # print(x.shape)
        # print(x[0].shape)
        #x = x.index_fill_(0, torch.tensor([i for i in range(128)], device="cuda"), 128)
        # print(x)

        x_layers, x_last = self.lstm(x)
        x1 = x_last[-1][-1]
        # print(x_last[-1].shape)

        x2 = x1.detach()
        # print(x2.shape)
        x2 = self.reward_predictor(x2).squeeze(1)
        # print(x2.shape)
        # print(x2.shape)
        x1 = self.decoder(x1)
        # print(x.shape)
        return x1, x2
