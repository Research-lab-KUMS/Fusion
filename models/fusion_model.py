import torch
from torch import nn
from models.mobilenet2 import MobileNetBase
from models.pretrained_models import PretrainedModel


class FusionModel(nn.Module):

    def __init__(self, n_frames, mode, activation=nn.ReLU6(), n_classes=27):
        super(FusionModel, self).__init__()

        self.n_frames = n_frames
        self.mode = mode
        self.classifier = None
        self.vision_classifier = None
        self.tactile_classifier = None

        tactile_block_depths = 1 * [16] + 2 * [24] + 3 * [32] + 4 * [64] + 3 * [96] + 3 * [160] + 1 * [320]
        tactile_strides = (1 * [1] + 1 * [2] + 1 * [1] + 1 * [2] + 2 * [1] + 1 * [2] + 3 * [1]
                           + 3 * [1] + 3 * [1] + 1 * [1])
        tactile_expansion_ratios = 1 * [1] + (len(tactile_block_depths) - 1) * [6]
        tactile_dropouts = (len(tactile_block_depths) - 2) * [0.0] + 2 * [0.0]

        self.tactile_model = MobileNetBase(blocks_depths=tactile_block_depths,
                                           blocks_strides=tactile_strides,
                                           exp_ratios=tactile_expansion_ratios,
                                           blocks_dropouts=tactile_dropouts,
                                           preconv_depth=32,
                                           in_channels=1)
        num_out_tactile = tactile_block_depths[-1] * 4

        self.vision_model, num_out_vision = PretrainedModel(n_frames=n_frames, model_name='mobilenet',
                                                            pretrained=True, freeze=False).get_model()
        self.vision_model.avgpool = nn.Identity()

        self.ave_pooling = nn.AdaptiveAvgPool2d(1)
        self._make_classifier(num_out_tactile, n_classes)
        self.activation = activation
        self.vis_3d_pooler = nn.MaxPool3d(kernel_size=(n_frames, 1, 1), stride=1)
        self.tac_3d_pooler = nn.MaxPool3d(kernel_size=(n_frames, 1, 1), stride=1)
        self.dropout = nn.Dropout(0.2)
        self.vis_dropout = nn.Dropout(0.2)
        self.tac_dropout = nn.Dropout(0.2)
        self.num_params = self.get_num_parameters()

    def forward(self, x_vision, x_tactile):
        vision_features, tactile_features = self.get_features(x_vision, x_tactile)

        if not len(vision_features.shape) > 1:
            vision_features = vision_features.unsqueeze(0)
            tactile_features = tactile_features.unsqueeze(0)

        if self.mode == 'early':
            z = self.early_fusion_forward(vision_features, tactile_features)
        else:
            z = self.late_fusion_forward(vision_features, tactile_features)
        return z

    @staticmethod
    def normalize(x):
        return x

    def get_features(self, x_vision, x_tactile):

        if self.n_frames > 1:
            vision_outputs = []
            tactile_outputs = []
            for i in range(self.n_frames):
                vis_frame_output = self.vision_model(x_vision[:, i])
                vision_outputs += [vis_frame_output]

                tac_frame_output = self.tactile_model(x_tactile[:, i])
                tactile_outputs += [tac_frame_output]

            vision_outputs = torch.stack(vision_outputs, dim=2)
            # vision_outputs = self.vision_conv1x1(vision_outputs)
            vision_outputs = self.vis_3d_pooler(vision_outputs)
            vision_outputs = self.ave_pooling(vision_outputs).squeeze()

            tactile_outputs = torch.stack(tactile_outputs, dim=2)
            # tactile_outputs = self.tactile_conv1x1(tactile_outputs)
            tactile_outputs = self.tac_3d_pooler(tactile_outputs)
            tactile_outputs = self.ave_pooling(tactile_outputs).squeeze()

        else:
            vision_outputs = self.vision_model(x_vision)
            vision_outputs = self.ave_pooling(vision_outputs).squeeze()

            tactile_outputs = self.tactile_model(x_tactile)
            tactile_outputs = self.ave_pooling(tactile_outputs).squeeze()

        return vision_outputs, tactile_outputs

    def early_fusion_forward(self, vision_features, tactile_features):
        z = torch.cat([vision_features, tactile_features], dim=1)
        z = self.dropout(z)
        z = self.classifier(z)
        return z

    def late_fusion_forward(self, vision_features, tactile_features):
        vision_features = self.vis_dropout(vision_features)
        tactile_features = self.tac_dropout(tactile_features)
        vision_preds = self.vision_classifier(vision_features)
        tactile_preds = self.tactile_classifier(tactile_features)

        vision_preds = torch.softmax(vision_preds, dim=1)
        tactile_preds = torch.softmax(tactile_preds, dim=1)
        z = torch.stack([vision_preds, tactile_preds], dim=2)
        z = torch.mean(z, dim=2)

        return z

    def _make_classifier(self, in_features, n_classes=27):
        if self.mode == 'early':
            self.classifier = nn.Linear(in_features * 2, n_classes)
        elif self.mode == 'late':
            self.vision_classifier = nn.Linear(in_features, n_classes)
            self.tactile_classifier = nn.Linear(in_features, n_classes)
        else:
            raise ValueError('Undefined mode...')

    def get_num_parameters(self):
        total_params = 0
        for param_name, param in self.named_parameters():
            if param.requires_grad is True:
                total_params += torch.numel(param)

        return int(total_params)


if __name__ == '__main__':
    N_FRAMES = 2
    random_vbatch = torch.randn(size=(2, N_FRAMES, 3, 200, 200), dtype=torch.float32)
    random_tbatch = torch.randn(size=(2, N_FRAMES, 1, 32, 32), dtype=torch.float32)
    model = FusionModel(n_frames=N_FRAMES, mode='late')
    print('Number of Learnable Parameters in Model =', model.num_params)
    preds = model(random_vbatch, random_tbatch)
    print('Output Shape of the Model:', preds.size())
    print(preds)
    print(torch.argmax(preds, dim=1))
