import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

from mobilenetv3 import MobileNetV3_Large
from bifpn import BiFPN


class CenterNet(nn.Module):
    def __init__(self, num_class, out_channels):
        super(CenterNet, self).__init__()
        self.deconv_with_bias = False
        self.heads = {'hm': num_class, 'wh': 2, 'reg': 2}
        head_conv = 64
        self.inplanes = 256
        self.mask_res_conv = nn.Conv2d(out_channels, 1 + out_channels, 1, 1, 0)

        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(out_channels, head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                          kernel_size=1, stride=1, padding=0))
            self.__setattr__(head, fc)

    def forward(self, x):
        out = self.mask_res_conv(x)
        mask = torch.sigmoid(out[:, 0:1])
        residual = torch.sigmoid(out[:, 1:]) * 2 - 1
        feature = mask * x + residual
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(feature)

        return ret, mask


class FastDet(nn.Module):
    def __init__(self, num_classes=2, out_channels=80, num_frames=1, output_shape=128, pretrained=None):
        super(FastDet, self).__init__()
        self.backbone = MobileNetV3_Large(out_channels, num_frames)
        self.bifpn = nn.Sequential(
            *[BiFPN(num_channels=out_channels,
                    attention=True)
              for _ in range(1)])
        self.merge_conv = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 1, 3, 1, 1, bias=True),
        )
        self.centernet = CenterNet(num_classes, out_channels)


        if pretrained:
            checkpoint = torch.load(pretrained)['state_dict']
            new_checkpoint = {}
            for key, val in checkpoint.items():
                key_split = key.split('.')[1:]
                if 'bneck' in key:
                    new_key = '.'.join([key_split[0] + key_split[1], *key_split[2:]])
                    new_checkpoint[new_key] = val
                elif 'bn1' in key:
                    new_key = '.'.join(key_split)
                    new_checkpoint[new_key] = val
                elif 'conv1' in key:
                    new_key = '.'.join(key_split)
                    new_checkpoint[new_key] = val.repeat(1, num_frames, 1, 1) / num_frames
                else:
                    pass
            res = self.backbone.load_state_dict(new_checkpoint, strict=False)
            print(res)


    def forward(self, x1, x2):
        out = self.backbone(x1)
        out = self.bifpn(out)
        x2 = self.merge_conv(x2)
        x2 = F.sigmoid(x2)
        out = out * x2 + out
        out, mask = self.centernet(out)
        return out, mask