# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model import flow_deform
from model import recons_video
from utils import utils
from model.blocks import Get_gradient


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    is_mask_filter = True
    return ERDN(in_channels=args.n_colors, n_sequence=args.n_sequence, out_channels=args.n_colors,
                n_resblock=args.n_resblock, n_feat=args.n_feat,
                is_mask_filter=is_mask_filter, device=device, phase=args.phase)


class ERDN(nn.Module):
    def __init__(self, in_channels=3, n_sequence=5, out_channels=3, freeze_flow=False, freeze_recons=False,
                 n_resblock=3, n_feat=32, is_mask_filter=True, device='cuda', phase='double'):
        super(ERDN, self).__init__()
        print("Model: Creating ERDN Net")

        self.n_sequence = n_sequence
        self.device = device
        self.phase = phase

        self.is_mask_filter = is_mask_filter
        extra_channels = 1

        # In this model, flow_net is used to estimate optical flow, while recons_net is used to reconstruction frames
        self.flow_net = flow_deform.Flow_Deform()
        self.recons_net = recons_video.RECONS_VIDEO(in_channels=in_channels, out_channels=out_channels,
                                                    n_resblock=n_resblock, n_feat=n_feat, extra_channels=extra_channels)

        self.grad_net = recons_video.RECONS_GRAD()
        self.get_grad = Get_gradient()

        self.flow_net_new = flow_deform.Flow_Deform()
        self.recons_net_new = recons_video.RECONS_VIDEO(in_channels=in_channels, out_channels=out_channels,
                                                        n_resblock=n_resblock, n_feat=n_feat, extra_channels=extra_channels)

        self.grad_net_new = recons_video.RECONS_GRAD()

        if freeze_flow:
            print('Freeze FlowNet')
            for param in self.flow_net.parameters():
                param.requires_grad = False

        if freeze_recons:
            print('Freeze ReconsNet')
            for param in self.recons_net.parameters():
                param.requires_grad = False
            for param in self.grad_net.parameters():
                param.requires_grad = False

    def get_masks(self, img_list):
        num_frames = len(img_list)

        img_list_copy = [img.detach() for img in img_list]  # detach backward
        if self.is_mask_filter:  # mean filter
            # assign mean filter to all the image
            img_list_copy = [utils.calc_meanFilter(im, n_channel=3, kernel_size=5) for im in img_list_copy]

        # use MSE to determine whether a pixel is sharp
        delta = 0.5
        mid_frame = img_list_copy[num_frames // 2]
        diff = torch.zeros_like(mid_frame)
        for i in range(num_frames):
            diff = diff + (img_list_copy[i] - mid_frame).pow(2)
        diff /= delta * delta
        diff = torch.sqrt(torch.sum(diff, dim=1, keepdim=True))
        luckiness = torch.exp(-diff)  # (0,1)

        return luckiness

    def first_stage(self, x):
        input_center = x[:, 1, :, :, :].clone()
        grad_input = self.get_grad(input_center)

        warped = self.flow_net(x)

        frame_warp_list = [warped[:, idx, :] for idx in range(warped.shape[1])]
        luckiness = self.get_masks(frame_warp_list)
        concated = torch.cat(frame_warp_list + [luckiness], dim=1)

        # extract feature
        scale_inblock = self.recons_net.inBlock(concated)
        scale_encoder_first = self.recons_net.encoder_first(scale_inblock)
        scale_encoder_second = self.recons_net.encoder_second(scale_encoder_first)
        feature = [scale_inblock, scale_encoder_first, scale_encoder_second]

        # reconstruct gradient map
        scale, shift, grad = self.grad_net(grad_input, feature)

        output = []
        for idx in range(len(feature)):
            feat = feature[idx]
            out = feat * scale[idx] + shift[idx]
            output.append(out)

        scale_decoder_second = self.recons_net.decoder_second(output[2])
        scale_decoder_first = self.recons_net.decoder_first(scale_decoder_second + output[1])
        out = self.recons_net.outBlock(scale_decoder_first + output[0])

        return out, grad

    def forward(self, x):
        frame_list = [x[:, i, :, :, :] for i in range(self.n_sequence)]

        if self.phase == 'single':
            warped2, grad2 = self.first_stage(torch.stack([frame_list[1], frame_list[2], frame_list[3]], dim=1))

            return warped2, grad2
        else:
            warped1, grad1 = self.first_stage(torch.stack([frame_list[0], frame_list[1], frame_list[2]], dim=1))
            warped2, grad2 = self.first_stage(torch.stack([frame_list[1], frame_list[2], frame_list[3]], dim=1))
            warped3, grad3 = self.first_stage(torch.stack([frame_list[2], frame_list[3], frame_list[4]], dim=1))
            warped_list = [warped1, warped2, warped3]

            grad_input = self.get_grad(warped2.clone())

            warped = self.flow_net_new(torch.stack(warped_list, dim=1))

            frame_warp_list = [warped[:, idx, :] for idx in range(warped.shape[1])]
            luckiness = self.get_masks(frame_warp_list)
            concated = torch.cat(frame_warp_list + [luckiness], dim=1)

            # extract feature
            scale_inblock = self.recons_net_new.inBlock(concated)
            scale_encoder_first = self.recons_net_new.encoder_first(scale_inblock)
            scale_encoder_second = self.recons_net_new.encoder_second(scale_encoder_first)
            feature = [scale_inblock, scale_encoder_first, scale_encoder_second]

            # reconstruct gradient map
            scale, shift, grad = self.grad_net_new(grad_input, feature)

            output = []
            for idx in range(len(feature)):
                feat = feature[idx]
                out = feat * scale[idx] + shift[idx]
                output.append(out)

            scale_decoder_second = self.recons_net_new.decoder_second(output[2])
            scale_decoder_first = self.recons_net_new.decoder_first(scale_decoder_second + output[1])
            out = self.recons_net_new.outBlock(scale_decoder_first + output[0])

            return out, grad
