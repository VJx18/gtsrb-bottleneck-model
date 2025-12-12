import torch.nn as nn

class LabelPredictor(nn.Module):
    def __init__(self, num_concepts=43, num_labels=43, dropout=0.3):

        super(LabelPredictor, self).__init__()

        # the layers: predict from concepts the labels
        self.linear_stack = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),      
            nn.Linear(128, num_labels)   # output: label logits
        )
    
    def forward(self, x):

        raw_logits = self.linear_stack(x)
        return raw_logits