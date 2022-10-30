from torch import nn


class ConvBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 groups=1,
                 batchnorm=True,
                 activation=nn.ReLU()):

        super(ConvBlock, self).__init__()
        layers = []
        conv_layer = nn.Conv2d(in_channels=inplanes,
                               out_channels=outplanes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=bias,
                               groups=groups)
        layers.append(conv_layer)
        if batchnorm is True:
            batch_norm = nn.BatchNorm2d(num_features=outplanes)
            layers.append(batch_norm)
        if activation is not None:
            layers.append(activation)

        self.composite = nn.Sequential(*layers)

    def forward(self, x):
        z = self.composite(x)
        return z


class SeparableConvBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 outplanes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 depthwise_activation=nn.ReLU(),
                 pointwise_activation=nn.ReLU()):
        super(SeparableConvBlock, self).__init__()

        depthwise_conv = ConvBlock(inplanes=inplanes,
                                   outplanes=inplanes,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=inplanes,
                                   activation=depthwise_activation)

        pointwise_conv = ConvBlock(inplanes=inplanes,
                                   outplanes=outplanes,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   groups=1,
                                   activation=pointwise_activation)

        layers = [depthwise_conv, pointwise_conv]
        self.composite = nn.Sequential(*layers)

    def forward(self, x):
        z = self.composite(x)
        return z
