import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
import torch
import torch.nn as nn
import torch.nn.init as init
import logging_functions as lf
import config as c


#################################################
#												#
#                 Components					#
#												#
#################################################


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


def init_network(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

def count_pars(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#################################################
#												#
#                 Models Zoo					#
#												#
#################################################
class BaselineCNN(nn.Module):
    def __init__(self, in_c, n_classes, slices=0.0):
        super().__init__()

        self.encoder = baseline_encoder(in_c, slices)

        self.decoder = baseline_decoder(n_classes)

    def forward(self, x):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)  # flat

        x = self.decoder(x)

        return x


class Resnet(nn.Module):
    def __init__(self, in_c, n_classes, layers=None):
        super().__init__()

        self.extractor = res_encoder(in_c, layers)

        self.classifier = baseline_decoder(n_classes, 64 * 3 * 2 * 2, 128)

    def forward(self, x):
        x = self.extractor(x)

        x = x.view(x.size(0), -1)  # flat

        x = self.classifier(x)

        return x

    
class DualResnet(nn.Module):
    def __init__(self, in_c, n_classes, slices=0.0):
        super().__init__()

        decoder_in = 64 * 3 * 1 * 2
        if slices > 0.0:
            decoder_in = c.dimension[slices]

        self.extractor1 = res_encoder(in_c, dual=True)

        self.extractor2 = res_encoder(in_c, dual=True)

        self.classifier = baseline_decoder(n_classes, decoder_in, 128)

        init_network(self.modules())

    def forward(self, x):

        x1, x2 = x[:, :, 0:64, :], x[:, :, 64:, :]

        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)

        x = torch.cat([x1, x2], dim=2)

        x = x.view(x.size(0), -1)  # flat

        x = self.classifier(x)

        return x


if __name__ == '__main__':
    import torch
    import numpy as np
    from torchinfo import summary
    from model_functions import count_pars

    layers = np.zeros(6, dtype=bool)
    # layers[1] = True

    # model = BaselineCNN(1, 10, 2)
    # model = DeformableCNN(1, 10, layers)
    # model = ExtendedCNN(3, 10, layers)

    model = DualResnet(3, 10, 5.0)
    # model = OverlapResnet(3, 10, layers, True, True)
    # print(model)
    # model_path = '../../../../nas/student/gPhD_Xin/workspace/dcase2020/models/2021-08-23-17-47-35_bs_16_lr_0.001_p_train_deformable/baseline.pth'
    # model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])

    batch_size, C, H, W = 16, 3, 128, 256
    x = torch.randn(batch_size, C, H, W)
    output = model(x)
    print(output.shape)

    summary(model, input_size=(batch_size, C, H, W))

    print(count_pars(model))
