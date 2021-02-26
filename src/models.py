import torch
import torch.nn as nn

import timm

class Network(nn.Module):
    """
    Currently I've only implmented 2 models (EfficientNet and ResNet15)
    
    Friendly warning: DONOT USE RESNET150, IT WILL PROBABLY RESULT IN AN OOM ERROR (OUT OF MEMORY)
    """
    def __init__(self, model_name='eff'):
        super(Network, self).__init__()
        self.model_name = model_name
        if self.model_name == 'eff':
            self.model = timm.create_model('efficientnet-b1', pretrained=True)
            self.model._fc = nn.Linear(in_features=1280, out_features=500, bias=True)
            self.output = nn.Linear(500, 1)
        elif self.model_name == 'res':
            self.model = timm.create_model('resnet18', pretrained=True)
            self.model.fc = nn.Linear(in_features=512, out_features=500, bias=True)
            self.output = nn.Linear(500, 1)
        
    def forward(self, img):
        img_feat = self.model(img)
        out = self.output(img_feat)
        return out