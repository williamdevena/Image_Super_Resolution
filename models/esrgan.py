import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.use_act=use_act

        self.cnn = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.act = nn.LeakyReLU(0.2) if self.use_act else nn.Identity()


    def forward(self, x):
        out = self.cnn(x)
        out = self.act(out)

        return out




class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True)
        self.act = nn.LeakyReLU(0.2)


    def forward(self, x):
        out = self.upsample(x)
        out = self.conv(out)
        out = self.act(out)

        return out



class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()


        for idx in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels=self.in_channels + self.channels*idx,
                    out_channels=self.channels if idx<=3 else self.in_channels,
                    use_act=True if idx <=3 else False
                )
            )


    def forward(self, x):
        new_inputs = x

        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.concat([new_inputs, out], dim=1)

        return self.residual_beta*out + x



class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.residual_beta = residual_beta

        self.rrdb = nn.Sequential([DenseResidualBlock(in_channels=self.in_channels)
                                    for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x)*self.residual_beta + x




class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=23):
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_blocks = num_blocks

        self.initial = nn.Conv2d(
            in_channels=self.in_channels
            out_channels=self.num_channels,
            kernel_size=9,
            stride=1,
            padding=1,
            bias=True
        )

        self.residuals = nn.Sequential([RRDB(in_channels=self.num_channels)
                                        for _ in range(self.num_blocks)])

        self.conv = nn.Conv2d(in_channels=self.num_channels,
                              out_channels=self.num_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        self.upsamples = nn.Sequential(
            UpSampleBlock(in_channels=self.num_channels),
            UpSampleBlock(in_channels=self.num_channels)
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.num_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            )
        )