import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from block.focalSpatial_Block import FocalSpatial
from block.GlobalLocal_Attention import AttentionFusion
from block.DFF_Module import DFF, ConvReshape
from block.InvertedResidualBlock import InvertedResidual
from block.SplitMultiScale import MultiScale

from ComparativeExperiments.MSCA_Net_main.Models.networks.msca_net import msca_net
from ComparativeExperiments.LGI_Net_main.LGI_Net import LGINET
from ComparativeExperiments.MHorUNet_main.models.MHorUNet import MHorunet
from ComparativeExperiments.H2Former.models.H2Former import Res34_Swin_MS
from ComparativeExperiments.UNETV2.UNet_v2 import UNetV2

__all__ = ['DPWithFocalSpatial', 'msca_net', 'LGINET', 'MHorunet', 'Res34_Swin_MS', 'UNetV2']


class HSBlock(nn.Module):
    '''
    替代3x3卷积
    '''

    def __init__(self, in_ch, s=4):
        '''
        特征大小不改变
        :param in_ch: 输入通道
        :param s: 分组数
        '''
        super(HSBlock, self).__init__()
        self.s = s
        self.module_list = nn.ModuleList()
        # 避免无法整除通道数
        in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
        self.module_list.append(nn.Sequential())
        acc_channels = 0
        for i in range(1, self.s):
            if i == 1:
                channels = in_ch
                acc_channels = channels // 2
            elif i == s - 1:
                channels = in_ch_last + acc_channels
            else:
                channels = in_ch + acc_channels
                acc_channels = channels // 2
            self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
        self.initialize_weights()

    def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        return conv_bn_relu

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = list(x.chunk(chunks=self.s, dim=1))
        for i in range(1, len(self.module_list)):
            y = self.module_list[i](x[i])
            if i == len(self.module_list) - 1:
                x[0] = torch.cat((x[0], y), 1)
            else:
                y1, y2 = y.chunk(chunks=2, dim=1)
                x[0] = torch.cat((x[0], y1), 1)
                x[i + 1] = torch.cat((x[i + 1], y2), 1)
        return x[0]


class ResEncoder_hs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder_hs, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = HSBlock(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = out + residual
        out = self.relu(out)
        return out


class UP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class DPWithFocalSpatial(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DPWithFocalSpatial, self).__init__()
        self.filters = [32, 64, 128, 256, 512]

        self.channels = in_channels

        self.downsample = nn.Sequential(
            nn.LayerNorm([self.filters[0], 224, 320]),
            nn.Conv2d(self.filters[0], self.filters[1], kernel_size=(2, 2), stride=(2, 2))
        )

        self.convnext = timm.create_model('convnextv2_pico', pretrained=True,
                                          pretrained_cfg_overlay=dict(file='model_pth/convnextv2_pico_1k_224_ema.pt'), )

        self.focal_spatial_enc_input = FocalSpatial(self.filters[0])
        self.focal_spatial_enc_1 = FocalSpatial(self.filters[1])
        self.focal_spatial_enc_2 = FocalSpatial(self.filters[2])
        self.focal_spatial_enc_3 = FocalSpatial(self.filters[3])
        self.focal_spatial_enc_4 = FocalSpatial(self.filters[4])

        self.enc_input = ResEncoder_hs(self.channels, self.filters[0])
        self.enc_1 = self.convnext.stages[0]
        self.enc_2 = self.convnext.stages[1]
        self.enc_3 = self.convnext.stages[2]
        self.enc_4 = self.convnext.stages[3]

        self.conv1x1_4 = nn.Conv2d(self.filters[4], self.filters[3], 1)
        self.conv1x1_3 = nn.Conv2d(self.filters[3], self.filters[2], 1)
        self.conv1x1_2 = nn.Conv2d(self.filters[2], self.filters[1], 1)
        self.conv1x1_1 = nn.Conv2d(self.filters[1], self.filters[0], 1)
        self.skip_connection_4 = AttentionFusion(self.filters[4])
        self.skip_connection_3 = AttentionFusion(self.filters[3])
        self.skip_connection_2 = AttentionFusion(self.filters[2])
        self.skip_connection_1 = AttentionFusion(self.filters[1])

        self.chunk_3 = MultiScale(self.filters[3])
        self.chunk_2 = MultiScale(self.filters[2])
        self.chunk_1 = MultiScale(self.filters[1])
        self.chunk_0 = MultiScale(self.filters[0])

        self.up_4 = UP(self.filters[4], self.filters[3])
        self.double_conv_4 = InvertedResidual(self.filters[4], self.filters[3])
        self.up_3 = UP(self.filters[3], self.filters[2])
        self.double_conv_3 = InvertedResidual(self.filters[3], self.filters[2])
        self.up_2 = UP(self.filters[2], self.filters[1])
        self.double_conv_2 = InvertedResidual(self.filters[2], self.filters[1])
        self.up_1 = UP(self.filters[1], self.filters[0])
        self.double_conv_1 = InvertedResidual(self.filters[1], self.filters[0])
        #
        self.conv_reshape_4 = ConvReshape(self.filters[3], 4, scale_factor=(224, 320))
        self.conv_reshape_3 = ConvReshape(self.filters[2], 4, scale_factor=(224, 320))
        self.conv_reshape_2 = ConvReshape(self.filters[1], 4, scale_factor=(224, 320))
        self.conv_reshape_1 = nn.Conv2d(self.filters[0], 4, kernel_size=1)
        #
        self.dff = DFF(16, 4)
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        enc_input_x = self.focal_spatial_enc_input(self.enc_input(x))

        enc_1x = self.focal_spatial_enc_1(self.enc_1(self.downsample(enc_input_x)))
        enc_2x = self.focal_spatial_enc_2(self.enc_2(enc_1x))
        enc_3x = self.focal_spatial_enc_3(self.enc_3(enc_2x))
        enc_4x = self.focal_spatial_enc_4(self.enc_4(enc_3x))

        combined_feature = enc_4x

        sc_4 = self.chunk_3(
            enc_3x + F.interpolate(self.conv1x1_4(self.skip_connection_4(enc_4x)),
                                   size=(enc_3x.shape[2], enc_3x.shape[3]), mode='bilinear'))  # 256

        sc_3 = self.chunk_2(
            enc_2x + F.interpolate(self.conv1x1_3(self.skip_connection_3(enc_3x)),
                                   size=(enc_2x.shape[2], enc_2x.shape[3]), mode='bilinear'))  # 128

        sc_2 = self.chunk_1(
            enc_1x + F.interpolate(self.conv1x1_2(self.skip_connection_2(enc_2x)),
                                   size=(enc_1x.shape[2], enc_1x.shape[3]), mode='bilinear'))  # 64
        sc_1 = self.chunk_0(
            enc_input_x + F.interpolate(self.conv1x1_1(self.skip_connection_1(enc_1x)),
                                        size=(enc_input_x.shape[2], enc_input_x.shape[3]), mode='bilinear'))  # 32

        decoder_4 = self.up_4(combined_feature)
        decoder_4 = self.double_conv_4(torch.cat([sc_4, decoder_4], dim=1))
        decoder_3 = self.up_3(decoder_4)
        decoder_3 = self.double_conv_3(torch.cat([sc_3, decoder_3], dim=1))
        decoder_2 = self.up_2(decoder_3)
        decoder_2 = self.double_conv_2(torch.cat([sc_2, decoder_2], dim=1))
        decoder_1 = self.up_1(decoder_2)
        decoder_1 = self.double_conv_1(torch.cat([sc_1, decoder_1], dim=1))
        #
        c_r_4 = self.conv_reshape_4(decoder_4)
        c_r_3 = self.conv_reshape_3(decoder_3)
        c_r_2 = self.conv_reshape_2(decoder_2)
        c_r_1 = self.conv_reshape_1(decoder_1)
        #
        feature_fusion = torch.cat([c_r_4, c_r_3, c_r_2, c_r_1], dim=1)
        feature_spatial = self.dff(feature_fusion)
        final = self.final(feature_spatial)

        return final



