import torch
import torch.nn as nn
import torchvision.models as models

from FeatureExtractor import IntermediateLayerGetter

Model_Dict ={
    'resnet': 'layer4',
    'densenet': 'features.layer4'
}
class PIRLModel(nn.Module):
    def __init__(self, net, ):

