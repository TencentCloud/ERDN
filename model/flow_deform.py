# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from model.flow_block import make_layer, ResidualBlockNoBN, DCNv2Pack
from model.recons_video import RECONS_VIDEO


class Flow_Deform(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_frame=3, deformable_groups=8,
                 num_extract_block=5, num_reconstruct_block=10, freeze=False):
        super(Flow_Deform, self).__init__()

        self.center_frame_idx = num_frame // 2

        # define deformable align network
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l4_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l4_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)

        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # define Encoder-Decoder Network
        self.center_net = RECONS_VIDEO()
        self.conv_l4_c_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l4_c_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # define 1*1 convolution layer
        self.conv_l1_t = nn.Conv2d(num_feat // 2, num_feat, 1, 1, 0)
        self.conv_l2_t = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.conv_l3_t = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)

        # freeze parameters of Encoder-Decoder Network
        if freeze:
            print('Freeze CenterNet')
            for param in self.center_net.parameters():
                param.requires_grad = False


    def forward(self, x):
        b, t, c, h, w = x.size()
        assert h % 8 == 0 and w % 8 == 0, 'The height and width must be multiple of 8.'

        x_neighbor = [x[:, 0, :], x[:, 2, :]]
        x_neighbor = torch.stack(x_neighbor, dim=1)

        x_center = [x[:, idx, :] for idx in range(t)]
        x_center = torch.cat(x_center, dim=1)

        # Extracting feature for neighboring frames
        feat_l1 = self.lrelu(self.conv_first(x_neighbor.view(-1, c, h, w)))
        feat_l1 = self.feature_extraction(feat_l1)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        # L4
        feat_l4 = self.lrelu(self.conv_l4_1(feat_l3))
        feat_l4 = self.lrelu(self.conv_l4_2(feat_l4))

        # Extracting feature for center frame
        first_scale_inblock = self.center_net.inBlock(x_center)
        feat_l1_center = self.conv_l1_t(first_scale_inblock)
        # L2
        first_scale_encoder_first = self.center_net.encoder_first(first_scale_inblock)
        feat_l2_center = self.conv_l2_t(first_scale_encoder_first)
        # L3
        first_scale_encoder_second = self.center_net.encoder_second(first_scale_encoder_first)
        feat_l3_center = self.conv_l3_t(first_scale_encoder_second)
        # L4
        feat_l4_center = self.lrelu(self.conv_l4_1(feat_l3_center))
        feat_l4_center = self.lrelu(self.conv_l4_2(feat_l4_center))

        feat_l1 = feat_l1.view(b, t - 1, -1, h, w)
        feat_l1 = torch.stack([feat_l1[:, 0, :], feat_l1_center, feat_l1[:, 1, :]], dim=1)

        feat_l2 = feat_l2.view(b, t - 1, -1, h // 2, w // 2)
        feat_l2 = torch.stack([feat_l2[:, 0, :], feat_l2_center, feat_l2[:, 1, :]], dim=1)

        feat_l3 = feat_l3.view(b, t - 1, -1, h // 4, w // 4)
        feat_l3 = torch.stack([feat_l3[:, 0, :], feat_l3_center, feat_l3[:, 1, :]], dim=1)

        feat_l4 = feat_l4.view(b, t - 1, -1, h // 8, w // 8)
        feat_l4 = torch.stack([feat_l4[:, 0, :], feat_l4_center, feat_l4[:, 1, :]], dim=1)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, self.center_frame_idx, :, :, :].clone(),
            feat_l2[:, self.center_frame_idx, :, :, :].clone(),
            feat_l3[:, self.center_frame_idx, :, :, :].clone(),
            feat_l4[:, self.center_frame_idx, :, :, :].clone()
        ]
        aligned_feat = []
        for i in range(t):
            if i != self.center_frame_idx:
                nbr_feat_l = [  # neighboring feature list
                    feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                    feat_l3[:, i, :, :, :].clone(), feat_l4[:, i, :, :, :].clone()
                ]
                aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))
        feat = torch.cat(aligned_feat, dim=0)  # (b*t, c, h, w)

        out = self.reconstruction(feat)
        out = self.conv_last(out)
        out = [out[b*i:b*i+b, :] for i in range(t-1)]

        # Generate reconstruction frame from feature of different scales
        first_scale_decoder_second = self.center_net.decoder_second(first_scale_encoder_second)
        first_scale_decoder_first = self.center_net.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
        first_scale_outBlock = self.center_net.outBlock(first_scale_decoder_first + first_scale_inblock)

        out.insert(self.center_frame_idx, first_scale_outBlock)
        out = torch.stack(out, dim=1)

        return out


class RFBlock(nn.Module):
    def __init__(self, level, num_feat):
        super(RFBlock, self).__init__()
        self.level = level

        block = [nn.Conv2d(num_feat * 2, num_feat, 1, 1, dilation=1),
                 nn.LeakyReLU(negative_slope=0.1),
                 nn.Conv2d(num_feat, num_feat, 3, 1, 1, dilation=1),
                 nn.LeakyReLU(negative_slope=0.1)]
        block = nn.Sequential(*block)

        self.rfblock = nn.ModuleList([block])

        for idx in range(level - 1):
            dilation = 2 ** (idx + 2) - 1
            kernel = dilation - 3

            block_1 = [nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, dilation=1),
                       nn.LeakyReLU(negative_slope=0.1)]
            while kernel > 0:
                block_1.extend([nn.Conv2d(num_feat, num_feat, 5, 1, 2, dilation=1),
                                nn.LeakyReLU(negative_slope=0.1)])
                kernel -= 4

            block_1.extend([nn.Conv2d(num_feat, num_feat, 3, 1, dilation, dilation=dilation),
                            nn.LeakyReLU(negative_slope=0.1, inplace=True)])
            block_1 = nn.Sequential(*block_1)
            self.rfblock.append(block_1)

        if level > 1:
            self.last_layer = nn.Conv2d(num_feat * len(self.rfblock), num_feat, 1, 1)

    def forward(self, x):
        if self.level > 1:
            output = []
            for block in self.rfblock:
                output.append(block(x))
            output = torch.cat(output, dim=1)
            output = self.last_layer(output)
        else:
            output = self.rfblock[0](x)

        return output


class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()

        # Pyramids
        for i in range(4, 0, -1):
            level = f'l{i}'
            if i == 4:
                self.offset_conv1[level] = RFBlock(level=1, num_feat=num_feat)
            elif i == 3:
                self.offset_conv1[level] = RFBlock(level=2, num_feat=num_feat)
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            elif i == 2:
                self.offset_conv1[level] = RFBlock(level=3, num_feat=num_feat)
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            elif i == 1:
                self.offset_conv1[level] = RFBlock(level=4, num_feat=num_feat)
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 4:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
        for i in range(4, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i != 4:
                offset = self.lrelu(self.offset_conv2[level](torch.cat([offset, upsampled_offset], dim=1)))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 4:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))

        return feat
