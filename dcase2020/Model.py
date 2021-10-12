import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.extractor = nn.Identity()
        self.classifier = nn.Identity()
        self.init_network()

    def forward(self, x):
        x = self.extractor(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def init_network(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def count_pars(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
