import torch
from torch import nn
from models.conv_basics import SeparableConvBlock, ConvBlock


class BottleneckBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride, exp_ratio, dropout=0.0):
        super(BottleneckBlock, self).__init__()

        expanded_channels = int(inplanes * exp_ratio)
        if exp_ratio > 1:
            self.expansion_conv = ConvBlock(inplanes=inplanes,
                                            outplanes=expanded_channels,
                                            kernel_size=1,
                                            padding=0,
                                            activation=nn.ReLU6())
        else:
            self.expansion_conv = nn.Identity()

        self.sep_conv = SeparableConvBlock(inplanes=expanded_channels,
                                           outplanes=outplanes,
                                           kernel_size=3,
                                           stride=stride,
                                           padding=1,
                                           depthwise_activation=nn.ReLU6(),
                                           pointwise_activation=None)

        self.use_res_connection = (stride == 1) and (inplanes == outplanes)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        z = self.expansion_conv(x)
        z = self.sep_conv(z)
        if self.use_res_connection:
            z = self.dropout(x + z)
        else:
            z = self.dropout(z)
        return z


class MobileNetBase(nn.Module):

    def __init__(self,
                 blocks_depths,
                 blocks_strides,
                 exp_ratios,
                 blocks_dropouts=None,
                 preconv_depth=32,
                 in_channels=1):

        super(MobileNetBase, self).__init__()
        self.preconv = ConvBlock(inplanes=in_channels,
                                 outplanes=preconv_depth,
                                 activation=nn.ReLU6())

        num_blocks = len(blocks_depths)
        n_input_channels = [preconv_depth] + blocks_depths[:-1]
        blocks = []
        for i in range(num_blocks):
            blocks.append(BottleneckBlock(inplanes=n_input_channels[i],
                                          outplanes=blocks_depths[i],
                                          stride=blocks_strides[i],
                                          exp_ratio=exp_ratios[i],
                                          dropout=blocks_dropouts[i]))

        self.blocks = nn.Sequential(*blocks)
        self.post_conv = ConvBlock(inplanes=blocks_depths[-1],
                                   outplanes=4 * blocks_depths[-1],
                                   kernel_size=1,
                                   padding=0,
                                   activation=nn.ReLU6())

    def forward(self, x):
        z = self.preconv(x)
        z = self.blocks(z)
        z = self.post_conv(z)
        return z


class MobileNetV2(nn.Module):

    def __init__(self,
                 n_frames,
                 blocks_depths,
                 blocks_strides,
                 exp_ratios,
                 blocks_dropouts=None,
                 preconv_depth=32,
                 in_channels=1,
                 n_classes=27):

        super(MobileNetV2, self).__init__()
        self.n_frames = n_frames
        self.base_model = MobileNetBase(blocks_depths=blocks_depths,
                                        blocks_strides=blocks_strides,
                                        blocks_dropouts=blocks_dropouts,
                                        in_channels=in_channels,
                                        exp_ratios=exp_ratios,
                                        preconv_depth=preconv_depth)

        self.local_pooler = nn.MaxPool3d(kernel_size=(n_frames, 1, 1), stride=1)
        self.global_pooler = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(blocks_depths[-1] * 4, n_classes)
        self.init_params()
        self.num_params = self.get_num_parameters()

    def forward(self, x):
        if self.n_frames > 1:
            branch_outputs = []
            for i in range(self.n_frames):
                z_frame = self.base_model(x[:, i])
                branch_outputs += [z_frame]
            z = torch.stack(branch_outputs, dim=2)
            # z = self.normalize(z)
            z = self.local_pooler(z).squeeze()
        else:
            z = self.base_model(x)

        z = self.global_pooler(z).squeeze()
        z = self.dropout(z)
        z = self.classifier(z)
        return z

    def get_num_parameters(self):
        total_params = 0
        for param_name, param in self.named_parameters():
            if param.requires_grad is True:
                total_params += torch.numel(param)
        return int(total_params)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.zeros_(m.bias)

    @staticmethod
    def normalize(x):
        x = x / torch.norm(x, dim=0, keepdim=True)
        return x


if __name__ == '__main__':
    N_FRAMES = 5
    random_batch = torch.randn(size=(32, N_FRAMES, 1, 32, 32), dtype=torch.float32)
    block_depths = 1 * [16] + 2 * [24] + 3 * [32] + 4 * [64] + 3 * [96] + 3 * [160] + 1 * [320]
    strides = 1 * [1] + 1 * [2] + 1 * [1] + 1 * [2] + 2 * [1] + 1 * [2] + 3 * [1] + 3 * [1] + 3 * [1] + 1 * [1]
    expansion_ratios = 1 * [1] + (len(block_depths) - 1) * [6]
    dropouts = (len(block_depths) - 2) * [0.0] + 2 * [0.0]

    model = MobileNetV2(n_frames=N_FRAMES,
                        blocks_depths=block_depths,
                        blocks_strides=strides,
                        exp_ratios=expansion_ratios,
                        blocks_dropouts=dropouts,
                        preconv_depth=32,
                        in_channels=1)

    print('Number of Learnable Parameters in Model =', model.num_params)
    preds = model(random_batch)
    print('Output Shape of the Model:', preds.size())
