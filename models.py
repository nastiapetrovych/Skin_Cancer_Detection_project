import torch
import torch.nn as nn
import timm

class ResNetBase(nn.Module):
    """
    ResNet-based model for binary classification.
    """
    def __init__(self, n_classes, pretrained=False, model_path=None):
        super(ResNetBase, self).__init__()
        self.model = timm.create_model("resnet50", pretrained=pretrained)
        self.name = "resnet50"

        if pretrained and model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        return self.model(x)
    

class EfficientNetBase(nn.Module):
    """
    EfficientNet-based model for binary classification.
    """
    def __init__(self, n_classes, pretrained=False, model_path=None):
        super(EfficientNetBase, self).__init__()
        self.model = timm.create_model("efficientnet_b0", pretrained=pretrained)
        self.name = "efficientnet_b0"

        if pretrained and model_path:
            self.model.load_state_dict(torch.load(model_path))

        self.model.classifier = nn.Linear(self.model.classifier.in_features, n_classes)

    def forward(self, x):
        return self.model(x)
