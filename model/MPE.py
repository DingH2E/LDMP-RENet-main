from torch import nn

class MPE(nn.Module):
    def __init__(self,in_channels=256, reduction_ratio=16):
        super(MPE, self).__init__()
        self.sSE = sSE(in_channels)
        self.cSE = cSE(in_channels, reduction_ratio)


    def forward(self, x):
        x = self.cSE(x)
        x = self.sSE(x)
        return x


class sSE(nn.Module):
    def __init__(self, in_channels=256):
        super(sSE, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        v = self.conv(x)
        v = self.norm(v)

        return v * x


class cSE(nn.Module):
    def __init__(self, in_channels=256, reduction_ratio=16):
        super(cSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


