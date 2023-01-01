'''MobileNetV3 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, out_channels=80, num_frames=2):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3 * num_frames, 16, kernel_size=3, stride=2, padding=1, bias=False)  # input 512 * 512, output 256x256
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck0 = Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1)  #output  256 x 256
        self.bneck1 = Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2)
        self.bneck2 = Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1)
        self.fuse1 = nn.Conv2d(24, out_channels, 1, bias=False)              #output  128 x 128

        self.bneck3 = Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2)
        self.bneck4 = Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1)
        self.bneck5 = Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1)
        self.fuse2 = nn.Conv2d(40, out_channels, 1, bias=False)              #output 64 x 64

        self.bneck6 = Block(3, 40, 240, 80, hswish(), None, 2)
        self.bneck7 = Block(3, 80, 200, 80, hswish(), None, 1)
        self.bneck8 = Block(3, 80, 184, 80, hswish(), None, 1)
        self.bneck9 = Block(3, 80, 184, 80, hswish(), None, 1)
        self.bneck10 = Block(3, 80, 480, 112, hswish(), SeModule(112), 1)
        self.bneck11 = Block(3, 112, 672, 112, hswish(), SeModule(112), 1)
        self.bneck12 = Block(5, 112, 672, 160, hswish(), SeModule(160), 1)
        self.fuse3 = nn.Conv2d(160, out_channels, 1, bias=False)           #output 32 x 32

        self.bneck13 = Block(5, 160, 672, 160, hswish(), SeModule(160), 2)
        self.bneck14 = Block(5, 160, 960, 160, hswish(), SeModule(160), 1)
        self.fuse4 = nn.Conv2d(160, out_channels, 1, bias=False)           #output 16 x 16

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    '''
    def forward(self, imgs):
        out_list = []
        out_1 = self.hs1(self.bn1(self.conv1(imgs[:, :3])))
        out_1 = self.bneck0(out_1)
        out_1 = self.bneck1(out_1)
        out_1 = self.bneck2(out_1)
        out_2 = self.hs1(self.bn1(self.conv1(imgs[:, 3:])))
        out_2 = self.bneck0(out_2)
        out_2 = self.bneck1(out_2)
        out_2 = self.bneck2(out_2)
        out = torch.cat([out_1, out_2], dim=1)
        out_list.append(self.fuse1(out))

        out_1 = self.bneck3(out_1)
        out_1 = self.bneck4(out_1)
        out_1 = self.bneck5(out_1)
        # out_2 = self.bneck3(out_2)
        # out_2 = self.bneck4(out_2)
        # out_2 = self.bneck5(out_2)
        # out = torch.cat([out_1, out_2], dim=1)
        out_list.append(self.fuse2(out_1))

        out_1 = self.bneck6(out_1)
        out_1 = self.bneck7(out_1)
        out_1 = self.bneck8(out_1)
        out_1 = self.bneck9(out_1)
        out_1 = self.bneck10(out_1)
        out_1 = self.bneck11(out_1)
        out_1 = self.bneck12(out_1)
        # out_2 = self.bneck6(out_2)
        # out_2 = self.bneck7(out_2)
        # out_2 = self.bneck8(out_2)
        # out_2 = self.bneck9(out_2)
        # out_2 = self.bneck10(out_2)
        # out_2 = self.bneck11(out_2)
        # out_2 = self.bneck12(out_2)
        # out = torch.cat([out_1, out_2], dim=1)
        out_list.append(self.fuse3(out_1))

        out_1 = self.bneck13(out_1)
        out_1 = self.bneck14(out_1)
        # out_2 = self.bneck13(out_2)
        # out_2 = self.bneck14(out_2)
        # out = torch.cat([out_1, out_2], dim=1)
        out_list.append(self.fuse4(out_1))
        return out_list
    '''


    def forward(self, x):
        out_list = []
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck0(out)
        out = self.bneck1(out)
        out = self.bneck2(out)
        out_list.append(self.fuse1(out))

        out = self.bneck3(out)
        out = self.bneck4(out)
        out = self.bneck5(out)
        out_list.append(self.fuse2(out))

        out = self.bneck6(out)
        out = self.bneck7(out)
        out = self.bneck8(out)
        out = self.bneck9(out)
        out = self.bneck10(out)
        out = self.bneck11(out)
        out = self.bneck12(out)
        out_list.append(self.fuse3(out))

        out = self.bneck13(out)
        out = self.bneck14(out)
        out_list.append(self.fuse4(out))
        return out_list


if __name__ == '__main__':
    net = MobileNetV3_Large().cuda().eval()
    x = torch.randn(1,3,512,512).cuda()
    import time
    from model_summary import get_model_summary
    print(get_model_summary(net, x))
    start_time = time.time()
    for i in range(1000):
        y = net(x)
    print(time.time() - start_time)
