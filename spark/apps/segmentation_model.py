from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import mobilenet_v2


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels)
        self.conv2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MobileNetUNet(nn.Module):
    def __init__(self, num_classes: int = 11):
        super().__init__()

        features = mobilenet_v2(weights=None).features
        self.stem = nn.Sequential(OrderedDict((str(i), features[i]) for i in [0, 1]))
        self.block1 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [2, 3]))
        self.block2 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [4, 5, 6]))
        self.block3 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [7, 8, 9, 10, 11, 12, 13]))
        self.block4 = nn.Sequential(OrderedDict((str(i), features[i]) for i in [14, 15, 16, 17]))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.dec4 = DecoderBlock(512 + 96, 256)
        self.dec3 = DecoderBlock(256 + 32, 128)
        self.dec2 = DecoderBlock(128 + 24, 64)
        self.dec1 = DecoderBlock(64 + 16, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        s1 = self.stem(x)
        s2 = self.block1(s1)
        s3 = self.block2(s2)
        s4 = self.block3(s3)
        s5 = self.block4(s4)

        bottleneck = self.bottleneck(s5)

        d4 = nn.functional.interpolate(bottleneck, size=s4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, s4], dim=1))

        d3 = nn.functional.interpolate(d4, size=s3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = nn.functional.interpolate(d3, size=s2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = nn.functional.interpolate(d2, size=s1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        logits = self.head(d1)
        return nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device, num_classes: int = 11) -> MobileNetUNet:
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = MobileNetUNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model
