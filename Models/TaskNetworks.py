import torch
import torch.nn as nn


class JigsawTask(nn.Module):
    def __init__(self, input_size, encoding_size, jigsaw_size):
        super(JigsawTask, self).__init__()
        if isinstance(jigsaw_size, int):
            self.jigsaw_size = jigsaw_size**2
        elif isinstance(jigsaw_size, (tuple, list)):
            assert len(jigsaw_size) == 2
            self.jigsaw_size = jigsaw_size[0]*jigsaw_size[1]
        self.encoding_size = encoding_size
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(
                                nn.Linear(input_size, encoding_size, bias=False),
                                nn.BatchNorm1d(encoding_size),
                                nn.LeakyReLU()
                                )
        self.fc2 = nn.Sequential(
                                nn.Linear(encoding_size*self.jigsaw_size, encoding_size, bias=False),
                                nn.BatchNorm1d(encoding_size),
                                nn.LeakyReLU()
                                )
    
    def forward(self, x):
        # Input size : [B*Jigsaw_size, Channels, Height, Width]
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        # Input size : [B*Jigsaw_size, Channels]
        x = self.fc1(x)
        # Input size : [B*Jigsaw_size, Encoding_size]
        x = x.view(-1, self.jigsaw_size*self.encoding_size)
        # Input size : [B, Jigsaw_size*Encoding_size]
        x = self.fc2(x)
        # Input size : [B, Encoding_size]
        return x

class GenericTask(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(GenericTask, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
                                nn.Linear(input_size, encoding_size, bias=False),
                                nn.BatchNorm1d(encoding_size),
                                nn.LeakyReLU()
                                )
    def forward(self, x):
        # Input size : [B, Channels, Height, Width]
        x = self.pool(x)
        x = x.squeeze(-1).squeeze(-1)
        # Input size : [B, Channels]
        x = self.fc(x)
        # Input size : [B, Encoding_size]
        return x

