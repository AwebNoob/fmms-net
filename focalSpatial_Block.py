import torch
import torch.nn as nn


class FocalSpatial(nn.Module):
    def __init__(self, out_channels, expansion_factor=6):
        super(FocalSpatial, self).__init__()
        self.out_channels = out_channels
        mid_channels = (self.out_channels * expansion_factor)
        self.conv5x5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv1x7 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3),
                                 groups=out_channels)
        self.conv7x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0),
                                 groups=out_channels)
        self.conv1x9 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4),
                                 groups=out_channels)
        self.conv9x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0),
                                 groups=out_channels)
        self.conv1x11 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 11), stride=(1, 1), padding=(0, 5),
                                  groups=out_channels)
        self.conv11x1 = nn.Conv2d(out_channels, out_channels, kernel_size=(11, 1), stride=(1, 1), padding=(5, 0),
                                  groups=out_channels)
        self.D_WiseConv = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            # nn.GELU(),
        )
        self.conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()

    def forward(self, x):
        short_cut = x
        conv5x5_out = self.conv5x5(x) + short_cut
        spatial_7x7 = self.conv7x1(self.conv1x7(conv5x5_out))
        spatial_9x9 = self.conv9x1(self.conv1x9(conv5x5_out))
        spatial_11x11 = self.conv11x1(self.conv1x11(conv5x5_out))
        focal_spatial = spatial_7x7 + spatial_9x9 + spatial_11x11 + short_cut
        d_Conv_out = self.D_WiseConv(focal_spatial)
        Conv_out = self.conv1x1(d_Conv_out)
        final = self.gelu(short_cut + Conv_out)
        # final = self.gelu(Conv_out) + short_cut
        return final

#
# from torchinfo import summary
#
# model = FocalSpatial(64).cuda()
# summary(model, input_size=(1, 64, 14, 14))
