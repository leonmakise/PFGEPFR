import torch
import torch.nn as nn
import torch.nn.functional as F


def define_G(input_dim, output_dim):
    
    netG = Generator(input_dim, output_dim)
    netG = torch.nn.DataParallel(netG).cuda()

    return netG

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.conv = nn.Sequential(
            # 128
            convblock(input_dim, 24, 3, 1, 1),
            InvertedResidual(24, 24, 1),
            InvertedResidual(24, 24, 1),
            # 64
            InvertedResidual(24, 48, 2),
            InvertedResidual(48, 48, 1),
            InvertedResidual(48, 48, 1),
            # 128
            deconvblock(48, 24, 2, 2, 0),
            InvertedResidual(24, 12, 1),
            InvertedResidual(12, 12, 1),
            InvertedResidual(12, 12, 1),
            convblock(12, 12, 3, 1, 1),
            nn.Conv2d(12, output_dim, 1, 1, 0)
        )

    def forward(self, x):
        x = self.conv(x)
        return torch.sigmoid(x)


# basic modules
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        expand_ratio = 1

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class convblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super(convblock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

class resblock(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super(resblock, self).__init__()

        self.conv = nn.Sequential(
            convblock(dim, dim, kernel_size, stride, padding),
            nn.Conv2d(dim, dim, kernel_size, stride, padding),
            nn.InstanceNorm2d(dim)
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = x + self.conv(x)
        y = self.relu(x)
        return y

class deconvblock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super(deconvblock, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        y = self.deconv(x)
        return y
