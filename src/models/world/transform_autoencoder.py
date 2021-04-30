import torch
import torch.nn as nn

from collections import OrderedDict

'''
Transform_Autoencoder Class

Train an autoencoder that accepts additional image transformative inputs
'''


class Transform_Autoencoder(nn.Module):
    # accepts input for number of transforms
    def __init__(self, num_transforms=0, frame_stacks=1):
        super(Transform_Autoencoder, self).__init__()

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

        self.reward_predictor = nn.Sequential(OrderedDict([
            ('reward_conv1', nn.Conv2d(64, 16, kernel_size=7)),
            ('reward_relu1', nn.ReLU()),
            ('reward_flat1', nn.Flatten()),
            ('reward_linear1', nn.Linear(256, 64)),
            ('reward_relu2', nn.ReLU()),
            ('reward_linear2', nn.Linear(64, 1)),
            ('reward_relu3', nn.LeakyReLU()),
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
        x1 = self.bottleneck(x)
        x2 = x1.detach()
        x2 = self.reward_predictor(x2)
        # print(x2.shape)
        x1 = self.decoder(x1)
        # print(x.shape)
        return x1, x2
