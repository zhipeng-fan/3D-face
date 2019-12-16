import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_conv_layer_num):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, base_conv_layer_num)
        self.down1 = down(base_conv_layer_num, base_conv_layer_num*2)
        self.down2 = down(base_conv_layer_num*2, base_conv_layer_num*4)
        self.down3 = down(base_conv_layer_num*4, base_conv_layer_num*8)
        self.down4 = down(base_conv_layer_num*8, base_conv_layer_num*8)
        self.up1 = up(base_conv_layer_num*16, base_conv_layer_num*4, False)
        self.up2 = up(base_conv_layer_num*8, base_conv_layer_num*2, False)
        self.up3 = up(base_conv_layer_num*4, base_conv_layer_num, False)
        self.up4 = up(base_conv_layer_num*2, base_conv_layer_num, False)
        self.outc = outconv(base_conv_layer_num, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class DoubleHeadUNet(nn.Module):
    def __init__(self, n_channels, n_classes, base_conv_layer_num):
        super(DoubleHeadUNet, self).__init__()
        self.inc = inconv(n_channels, base_conv_layer_num)
        self.down1 = down(base_conv_layer_num, base_conv_layer_num*2)
        self.down2 = down(base_conv_layer_num*2, base_conv_layer_num*4)
        self.down3 = down(base_conv_layer_num*4, base_conv_layer_num*8)
        self.down4 = down(base_conv_layer_num*8, base_conv_layer_num*8)
        # Shared upsampling layer
        self.up1 = up(base_conv_layer_num*16, base_conv_layer_num*4, False)
        self.up2 = up(base_conv_layer_num*8, base_conv_layer_num*2, False)
        # Head dependent upsampling layer

        self.up3_a = up(base_conv_layer_num*4, base_conv_layer_num, False)
        self.up4_a = up(base_conv_layer_num*2, base_conv_layer_num, False)
        self.out_a = outconv(base_conv_layer_num, n_classes)
        
        self.up3_b = up(base_conv_layer_num*4, base_conv_layer_num, False)
        self.up4_b = up(base_conv_layer_num*2, base_conv_layer_num, False)
        self.out_b = outconv(base_conv_layer_num, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x_split = self.up2(x, x3)
        # First head
        x_a = self.up3_a(x_split, x2)
        x_a = self.up4_a(x_a, x1)
        x_a = self.out_a(x_a)
        # Second head
        x_b = self.up3_b(x_split, x2)
        x_b = self.up4_b(x_b, x1)
        x_b = self.out_b(x_b)
        return x_a, x_b


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(1,3,224,224)
    x = torch.Tensor(x)
    model = DoubleHeadUNet(3,3,64)
    a, b = model(x)
    print (a.shape, b.shape)