"""Convolutional encoding of persistence images."""

import torch.nn as nn


class TopologyEncoder(nn.Module):
    """Encode one-channel persistence images into feature vectors."""

    def __init__(self, dim_out: int, pretrained: bool = False):
        super().__init__()
        self.dim_out = dim_out

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, dim_out, 3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, persistence_image):
        """Return one feature vector per persistence image."""
        x = self.features(persistence_image)
        return x.view(x.size(0), -1)

    def reset_parameters(self):
        """Reset convolutional and normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


CNN_TopoEncoder = TopologyEncoder
