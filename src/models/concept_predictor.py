import torch
import torch.nn as nn
from torchvision import models

class ConceptPredictor(nn.Module):
    def __init__(self, num_concepts=15, dropout=0.3):
        super(ConceptPredictor, self).__init__()

        # using efficientnet_v2_s as backbone
        self.backbone = models.efficientnet_v2_s(weights='DEFAULT')

        # get the input size for the fc layer
        # efficientnet_v2_s has 1280 features
        backbone_out = 1280

        # remove the original classifier
        self.backbone.classifier = nn.Identity()

        # add our concept prediction layers
        self.fc1 = nn.Linear(backbone_out, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_concepts)  # output: concept logits

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # return logits
