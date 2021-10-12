import os
import sys
sys.path.append(os.path.join(sys.path[0], '../'))
import torch.nn as nn


# Note: this is easy to transplant
def conv_block(in_f, out_f, activation='relu', bn_learn=True, *args, **kwargs):
    activations = nn.ModuleDict({
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'none': nn.Identity()
    })

    conv = nn.Conv2d

    return nn.Sequential(
        conv(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f, affine=bn_learn),
        activations[activation],
    )


def baseline_encoder(in_c, slices=0.0):
    k = 100
    if slices > 0.0:
        k = int(slices * 10)
    encoder = nn.Sequential(
        conv_block(in_c, 16, kernel_size=7, padding=3),

        conv_block(16, 16, kernel_size=7, padding=3),
        nn.MaxPool2d(kernel_size=5),
        nn.Dropout(p=0.3),

        conv_block(16, 32, kernel_size=7, padding=3),
        nn.MaxPool2d(kernel_size=[4, k]),
        nn.Dropout(p=0.3)
    )
    return encoder


def res_encoder(in_c, layers=None, dual=False):

    res_block_bn_learn = True
    encoder = nn.Sequential(
        conv_block(in_c, 16, bn_learn=True, kernel_size=3, padding=1, stride=[1, 3]),

        ResNetBlock(16, 32, res_block_bn_learn),
        nn.MaxPool2d(kernel_size=1 if dual else 4),

        ResNetBlock(32, 32, res_block_bn_learn),
        nn.MaxPool2d(kernel_size=2),

        ResNetBlock(32, 64, res_block_bn_learn),
        nn.MaxPool2d(kernel_size=[4, 3]),

        ResNetBlock(64, 64, res_block_bn_learn),
        nn.MaxPool2d(kernel_size=8 if dual else [1, 2]),
        nn.Dropout(p=0.5),

    )
    return encoder


def baseline_decoder(n_classes, n_in=64, fc1_out=100):
    encoder = nn.Sequential(
        nn.Linear(n_in, fc1_out),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(fc1_out, n_classes)
    )
    return encoder


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn_learn=True):
        super(ResNetBlock, self).__init__()

        self.conv_identity = conv_block(in_ch, out_ch, activation='relu', bn_learn=bn_learn, kernel_size=1, padding=0, stride=1)
        self.conv1 = conv_block(in_ch, out_ch, bn_learn=bn_learn, kernel_size=3, padding=1, stride=1)
        self.conv2 = conv_block(out_ch, out_ch, bn_learn=bn_learn, kernel_size=3, padding=1, stride=1)


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)

        x += identity

        return x