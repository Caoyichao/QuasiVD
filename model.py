import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )


class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResBlock, self).__init__()
        if in_planes == out_planes and stride == 1:
            self.conv0 = nn.Identity()
        else:
            self.conv0 = nn.Conv2d(in_planes, out_planes,
                                   3, stride, 1, bias=False)
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv_wo_act(out_planes, out_planes, 3, 1, 1)
        self.relu1 = nn.PReLU(1)
        self.relu2 = nn.PReLU(out_planes)
        self.fc1 = nn.Conv2d(out_planes, 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(16, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu1(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = self.relu2(x * w + y)
        return x


class FFBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(FFBlock, self).__init__()
        self.res0 = ResBlock(in_channel, out_channel, stride)
        self.res1 = ResBlock(out_channel, out_channel)
        self.res2 = ResBlock(out_channel, out_channel)
        self.res3 = ResBlock(out_channel, out_channel)
    def forward(self, x):
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return x

class FFNet(nn.Module):  # as backbone
    def __init__(self):
        super(FFNet, self).__init__()
        self.conv0 = conv(6, 64, 3, 2, 1)
        self.block0 = FFBlock(64, 64, 2)
        self.block1 = FFBlock(64, 96, 1)
        self.block2 = FFBlock(96, 128, 1)
        self.conv1 = conv_wo_act(128, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 1 + 128, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        feature = self.conv1(x)
        out = self.conv2(x)
        mask = torch.sigmoid(out[:, 0:1])
        residual = torch.sigmoid(out[:, 1:]) * 2 - 1
        return feature, mask, residual


class DNet(nn.Module):
    def __init__(self, num_class):
        super(DNet, self).__init__()
        self.down0 = ResBlock(128, 256, 2)
        self.down1 = ResBlock(256, 512, 2)
        self.down2 = ResBlock(512, 1024, 2)
        self.up0 = deconv(1024, 512)
        self.up1 = deconv(512, 256)
        self.up2 = deconv(256, 128)
        self.heads = {'hm': num_class, 'wh': 2, 'reg': 2}
        for head in sorted(self.heads):
            num_output = self.heads[head]
            fc = nn.Conv2d(128, num_output, 3, 1, 1)
            self.__setattr__(head, fc)

    def forward(self, x):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


class CNet(nn.Module):
    def __init__(self, num_class=1):
        super(CNet, self).__init__()
        self.conv0 = conv(128, 256, 3, 2, 1)
        self.conv1 = conv(256, 512, 3, 2, 1)
        self.conv2 = conv(512, 1024, 3, 2, 1)
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class Model(nn.Module):
    def __init__(self, num_class=1):
        super(Model, self).__init__()
        self.backbone = FFNet()
        # self.cnet = CNet(num_class + 1)
        self.head = DNet(num_class)

    def forward(self, x):
        feature, mask, residual = self.backbone(x)
        # out = feature * mask + residual
        out = self.head(feature)
        # out = self.cnet(out)
        return out, mask



if __name__ == '__main__':
    model = Model(num_class=1)
    dummy_input = torch.randn(1, 6, 512, 512)
    out = model(dummy_input, is_train=False)
    for i in out:
        print(out[i].shape)
