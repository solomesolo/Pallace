import torch.nn as nn
from torchvision import  models


class PretrainedDensenet(nn.Module):
<<<<<<< HEAD
    def __init__(self, num_class=1):
=======
    def __init__(self, num_class=1, dropout_prob=0):
>>>>>>> first-stage
        super().__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=4)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
<<<<<<< HEAD
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
=======
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        
>>>>>>> first-stage
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         x = self.conv1(x)
        features = self.features(x)
<<<<<<< HEAD
        out = self.relu(features)
=======
        out = self.dropout_layer(features)
        out = self.relu(out)
        # dropout layer on convolution output
>>>>>>> first-stage
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        return self.sigmoid(self.fc1(out))
