import torch
import torch.nn as nn
from torchvision import models

class FaceIDModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.mobilenet_v2(pretrained=False) #not downloading pretrained weights
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

def load_model(num_classes, weights_path=None, device='cpu'):
    model = FaceIDModel(num_classes)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device) 