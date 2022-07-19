import torch.nn as nn
import torch
import model.blocks as blocks


def make_model(args):
    return RECONS_VIDEO(in_channels=args.n_colors,
                        n_sequence=args.n_sequence,
                        out_channels=args.n_colors,
                        n_resblock=args.n_resblock,
                        n_feat=args.n_feat)


class RECONS_VIDEO(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 kernel_size=5, extra_channels=0, feat_in=False):
        super(RECONS_VIDEO, self).__init__()

        self.feat_in = feat_in

        InBlock = []

        InBlock.extend([nn.Sequential(
            nn.Conv2d(in_channels * n_sequence + extra_channels, n_feat, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )])

        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                        for _ in range(n_resblock)])

        # encoder1
        Encoder_first = [nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_first.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                              for _ in range(n_resblock)])
        # encoder2
        Encoder_second = [nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )]
        Encoder_second.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                               for _ in range(n_resblock)])

        # decoder2
        Decoder_second = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)]
        Decoder_second.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))
        # decoder1
        Decoder_first = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                         for _ in range(n_resblock)]
        Decoder_first.append(nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        ))

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]
        OutBlock.append(
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

        self.inBlock = nn.Sequential(*InBlock)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.outBlock = nn.Sequential(*OutBlock)

    def forward(self, x):
        if x.ndimension() == 5:
            b, n, c, h, w = x.size()
            frame_list = [x[:, i, :, :, :] for i in range(n)]
            x = torch.cat(frame_list, dim=1)

        first_scale_inblock = self.inBlock(x)
        first_scale_encoder_first = self.encoder_first(first_scale_inblock)
        first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
        first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)
        first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)
        first_scale_outBlock = self.outBlock(first_scale_decoder_first + first_scale_inblock)

        mid_loss = None

        return first_scale_outBlock, mid_loss


class RECONS_GRAD(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_resblock=2, n_feat=32, kernel_size=3):
        super(RECONS_GRAD, self).__init__()

        # input layer
        self.in_layer = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1),
            blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1),
            blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1),
            blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1),
        )

        # input block
        InBlock = []
        InBlock.extend([blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                        for _ in range(n_resblock)])
        InBlock.append(nn.Conv2d(n_feat * 2, n_feat, kernel_size=kernel_size, padding=kernel_size // 2))
        self.inBlock = nn.Sequential(*InBlock)

        # encoder1
        self.down_first = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )
        Encoder_first = []
        Encoder_first.extend([blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                              for _ in range(n_resblock)])
        Encoder_first.append(nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=kernel_size, padding=kernel_size // 2))
        self.encoder_first = nn.Sequential(*Encoder_first)

        # encoder2
        self.down_second = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat * 4, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(inplace=True)
        )
        Encoder_second = []
        Encoder_second.extend([blocks.ResBlock(n_feat * 8, n_feat * 8, kernel_size=kernel_size, stride=1)
                               for _ in range(n_resblock)])
        Encoder_second.append(nn.Conv2d(n_feat * 8, n_feat * 4, kernel_size=kernel_size, padding=kernel_size // 2))
        self.encoder_second = nn.Sequential(*Encoder_second)

        # decoder2
        Decoder_second = [blocks.ResBlock(n_feat * 4, n_feat * 4, kernel_size=kernel_size, stride=1)
                          for _ in range(n_resblock)]
        self.decoder_second_up = nn.Sequential(
            nn.ConvTranspose2d(n_feat * 4, n_feat * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_second = nn.Sequential(*Decoder_second)

        # decoder1
        Decoder_first = [blocks.ResBlock(n_feat * 2, n_feat * 2, kernel_size=kernel_size, stride=1)
                         for _ in range(n_resblock)]
        self.decoder_first_up = nn.Sequential(
            nn.ConvTranspose2d(n_feat * 2, n_feat, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder_first = nn.Sequential(*Decoder_first)

        # output block
        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]
        self.outBlock = nn.Sequential(*OutBlock)

        # define recons layer
        self.recons_output = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        # define scale layer and shift layer for SFT
        self.condition_scale = nn.ModuleList()
        self.condition_shift = nn.ModuleList()
        for i in range(3):
            out_channels = n_feat * (2**i)
            self.condition_scale.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)))
            self.condition_shift.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1)))

    def forward(self, x, feat):
        condition_scale = []
        condition_shift = []

        # extract feature
        inlayer = self.in_layer(x)
        inlayer = torch.cat([inlayer, feat[0]], dim=1)
        inblock = self.inBlock(inlayer)

        down_first = self.down_first(inblock)
        down_first = torch.cat([down_first, feat[1]], dim=1)
        encoder_first = self.encoder_first(down_first)

        down_second = self.down_second(encoder_first)
        down_second = torch.cat([down_second, feat[2]], dim=1)
        encoder_second = self.encoder_second(down_second)

        # recons gradient map
        decoder_second = self.decoder_second(encoder_second)
        decoder_second_up = self.decoder_second_up(decoder_second)

        decoder_first = self.decoder_first(decoder_second_up + encoder_first)
        decoder_first_up = self.decoder_first_up(decoder_first)

        outblock = self.outBlock(decoder_first_up + inblock)

        feature = [outblock, decoder_first, decoder_second]
        # generate scale, shift and gradient map
        for idx in range(3):
            feat = feature[idx]

            scale = self.condition_scale[idx](feat)
            shift = self.condition_shift[idx](feat)

            condition_scale.append(scale.clone())
            condition_shift.append(shift.clone())

        gradient_map = self.recons_output(outblock)

        return condition_scale, condition_shift, gradient_map
