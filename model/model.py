import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import timm


class EfficientNetModel(BaseModel):
    def __init__(self, num_classes, model):
        super().__init__()
        self.backbone = timm.create_model(
            model, pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        x = self.backbone(x)
        return F.sigmoid(x)
