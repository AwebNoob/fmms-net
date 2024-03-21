import torch
import torch.nn as nn


class ConvReshape(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super(ConvReshape, self).__init__()
        self.conv_reshape = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Upsample(size=scale_factor, mode='bilinear'),
        )

    def forward(self, x):
        return self.conv_reshape(x)


class Group_Spatial_Attention(nn.Module):
    def __init__(self, channels, expand_factor):
        super(Group_Spatial_Attention, self).__init__()
        self.channels = channels
        self.expand_factor = expand_factor
        self.mid_channels = self.channels * self.expand_factor

        self.group_conv = nn.Sequential(
            nn.Conv2d(self.channels, self.mid_channels, kernel_size=3, stride=1, padding=1, groups=self.channels),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
        )

        # self.group_conv_1 = nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=2, groups=1)
        # self.group_conv_2 = nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3, groups=1)
        # self.group_conv_3 = nn.Conv2d(4, 1, kernel_size=9, stride=1, padding=4, groups=1)

        self.group_conv_1x5 = nn.Conv2d(16, 16, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.group_conv_5x1 = nn.Conv2d(16, 1, kernel_size=(5, 1), stride=1, padding=(2, 0))

        self.group_conv_1x7 = nn.Conv2d(16, 16, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.group_conv_7x1 = nn.Conv2d(16, 1, kernel_size=(7, 1), stride=1, padding=(3, 0))

        self.group_conv_1x9 = nn.Conv2d(16, 16, kernel_size=(1, 9), stride=1, padding=(0, 4))
        self.group_conv_9x1 = nn.Conv2d(16, 1, kernel_size=(9, 1), stride=1, padding=(4, 0))

        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x

        group_conv = self.group_conv(x)

        conv_split = torch.split(group_conv, self.mid_channels // self.expand_factor, dim=1)

        group_1 = sum((self.group_conv_5x1(self.group_conv_1x5(conv_split[i])) for i in range(len(conv_split))))
        group_2 = sum((self.group_conv_7x1(self.group_conv_1x7(conv_split[i])) for i in range(len(conv_split))))
        group_3 = sum((self.group_conv_9x1(self.group_conv_1x9(conv_split[i])) for i in range(len(conv_split))))
        group = group_1 + group_2 + group_3
        weighted = self.gelu(group)

        out = shortcut * weighted + shortcut
        return out


class L_Channel(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(L_Channel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DFF(nn.Module):
    def __init__(self, channels, expand_factor):
        super(DFF, self).__init__()
        self.l_channel = L_Channel(channels)
        self.group_s = Group_Spatial_Attention(channels, expand_factor)

    def forward(self, x):
        shortcut = x

        l_score = self.l_channel(x) * x + shortcut
        gs_score = self.group_s(l_score) + shortcut

        # group_att = self.group_s(x)

        out = gs_score
        return out

