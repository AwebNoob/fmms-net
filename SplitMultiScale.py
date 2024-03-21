import torch
import torch.nn as nn


class MultiScale(nn.Module):
    def __init__(self, in_channels):
        super(MultiScale, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=5, stride=1, padding=2)
        self.conv_3 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=7, stride=1, padding=3)
        self.conv_4 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=9, stride=1, padding=4)

        self.conv1x1 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1)

        self.gelu = nn.GELU()

    def forward(self, x):
        a, b, c, d = torch.chunk(x, chunks=4, dim=1)

        a3 = self.gelu(self.conv_1(a))
        a1 = self.gelu(self.conv1x1(a))
        a_final = a3 + a1

        b3 = self.gelu(self.conv_2(b + a_final))
        b1 = self.gelu(self.conv1x1(b + a_final))
        b_final = b3 + b1

        c3 = self.gelu(self.conv_3(c + b_final))
        c1 = self.gelu(self.conv1x1(c + b_final))
        c_final = c3 + c1

        d3 = self.gelu(self.conv_4(d + c_final))
        d1 = self.gelu(self.conv1x1(d + c_final))
        d_final = d3 + d1

        out = torch.cat([a_final, b_final, c_final, d_final], dim=1)
        return out



