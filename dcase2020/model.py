import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], '../'))
import Model as M
import model_components as mc
import constants as c

# Note: this is easy to transplant
class BaselineCNN(M.Model):
    def __init__(self, in_c, n_cls, slices=0.0):
        super(BaselineCNN, self).__init__()

        self.extractor = mc.baseline_encoder(in_c, slices)
        self.classifier = mc.baseline_decoder(n_cls)


class Resnet(M.Model):
    def __init__(self, in_c, n_classes, layers=None):
        super(Resnet, self).__init__()

        self.extractor = mc.res_encoder(in_c, layers)

        self.classifier = mc.baseline_decoder(n_classes, 64 * 3 * 2 * 2, 128)


class DualResnet(M.Model):
    def __init__(self, in_c, n_classes, slices=0.0):
        super(DualResnet, self).__init__()

        decoder_in = 64 * 3 * 1 * 2
        if slices > 0.0:
            decoder_in = c.dimension[slices]

        self.extractor1 = mc.res_encoder(in_c, dual=True)

        self.extractor2 = mc.res_encoder(in_c, dual=True)

        self.classifier = mc.baseline_decoder(n_classes, decoder_in, 128)

    def forward(self, x):
        x1, x2 = x[:, :, 0:64, :], x[:, :, 64:, :]

        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)

        x = torch.cat([x1, x2], dim=2)

        x = x.view(x.size(0), -1)  # flat

        x = self.classifier(x)

        return x


if __name__ == '__main__':
    import numpy as np
    from torchinfo import summary

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
