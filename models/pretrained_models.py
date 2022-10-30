import torch
import torchvision
from torch import nn
from models.pytorch_mobilenet import mobilenet_v2


class PretrainedModel(nn.Module):

    def __init__(self,
                 n_frames,
                 model_name='mobilenet',
                 freeze=False,
                 pretrained=True,
                 n_classes=27):

        super(PretrainedModel, self).__init__()
        self.n_frames = n_frames
        self.model_name = model_name
        self.freeze = freeze
        self.pretrained = pretrained
        self.base_model, self.fc_in_features = self.get_model()
        self.dropout = nn.Dropout(0.2)
        self.local_pooler = nn.MaxPool3d(kernel_size=(n_frames, 1, 1), stride=1)
        self.global_pooler = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.fc_in_features, n_classes)
        self.num_params = self.get_num_parameters()

    def forward(self, x):
        if self.n_frames > 1:
            branch_outputs = []
            for i in range(self.n_frames):
                branch_outputs += [self.base_model(x[:, i])]
            z = torch.stack(branch_outputs, dim=2)
            z = self.local_pooler(z).squeeze()
        else:
            z = self.base_model(x)

        z = self.global_pooler(z).squeeze()
        z = self.dropout(z)
        z = self.classifier(z)
        return z

    def get_model(self):
        if self.model_name == 'mobilenet':
            model = mobilenet_v2(pretrained=self.pretrained)
            fc_size = model.classifier.state_dict()['1.weight'].shape[-1]
            model.classifier = nn.Identity()
        elif self.model_name == 'resnet':
            model = torchvision.models.resnet18(pretrained=self.pretrained)
            fc_size = model.fc.weight.shape[-1]
            model.fc = nn.Identity()
        elif self.model_name == 'densenet':
            model = torchvision.models.densenet121(pretrained=self.pretrained)
            fc_size = model.classifier.weight.shape[-1]
            model.classifier = nn.Identity()
        elif self.model_name == 'shufflenet':
            model = torchvision.models.shufflenet_v2_x1_0(pretrained=self.pretrained)
            fc_size = model.fc.weight.shape[-1]
            model.fc = nn.Identity()
        elif self.model_name == 'resnext':
            model = torchvision.models.resnext50_32x4d(pretrained=self.pretrained)
            fc_size = model.fc.weight.shape[-1]
            model.fc = nn.Identity()
        elif self.model_name == 'wideresnet':
            model = torchvision.models.wide_resnet50_2(pretrained=self.pretrained)
            fc_size = model.fc.weight.shape[-1]
            model.fc = nn.Identity()
        elif self.model_name == 'mnasnet':
            model = torchvision.models.mnasnet1_0(pretrained=self.pretrained)
            fc_size = model.classifier.state_dict()['1.weight'].shape[-1]
            model.classifier = nn.Identity()
        else:
            raise ValueError('Undefined model name: "{}"'.format(self.model_name))

        if self.freeze:
            for module in model.modules():
                if isinstance(module, torch.nn.Linear) is not True:
                    module.requires_grad_(False)

        return model, fc_size

    def get_num_parameters(self):
        total_params = 0
        for param_name, param in self.named_parameters():
            if param.requires_grad is True:
                total_params += torch.numel(param)
        return int(total_params)


if __name__ == '__main__':
    N_FRAMES = 5
    random_batch = torch.randn(size=(32, N_FRAMES, 3, 200, 200), dtype=torch.float32)
    model = PretrainedModel(n_frames=N_FRAMES, model_name='mobilenet', freeze=False)
    print('Number of Learnable Parameters in Model =', model.num_params)
    preds = model(random_batch)
    print('Output Shape of the Model:', preds.size())
