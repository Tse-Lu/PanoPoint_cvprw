import torch
import torch.nn as nn
import torch.nn.functional as F


class BL1(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.conv(x)
        out = self.bn(out)

        return self.act(x + out)


class BL2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            BL1(32),
            BL2(32, 64)
        )

        self.layer2 = nn.Sequential(
            BL2(64, 128)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class DetectionHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(128, 65, 3, 1, 1),
            nn.BatchNorm2d(65),
            nn.ELU(inplace=True),

            nn.Conv2d(65, 65, 3, 1, 1)
        )

    def forward(self, x):
        x = self.head(x)  

        return x


class DescriptionHead(nn.Module):
    def __init__(self, upsample=True):
        super().__init__()

        self.upsample = upsample

        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ELU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, 1)
        )

    def forward(self, x):
        desc = self.head(x)

        if self.upsample:
            desc = F.interpolate(
                desc,
                scale_factor=8,
                mode='bilinear',
                align_corners=False
            )

        desc = F.normalize(
            desc,
            p=2,
            dim=1
        )

        return desc


class BaseDetector(nn.Module):
    def __init__(self, upsample=True):
        super().__init__()

        self.backbone = Encoder()
        self.detector = DetectionHead()
        self.descriptor = DescriptionHead(upsample=upsample)

    def forward(self, x):
        feat = self.backbone(x)
        prob = self.detector(feat)
        desc = self.descriptor(feat)

        return prob, desc


if __name__ == "__main__":
    model = BaseDetector()

    x = torch.randn(2, 1, 480, 960)
    prob, desc = model(x)

    print(prob.shape, desc.shape)
