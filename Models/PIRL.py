import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .FeatureExtractor import IntermediateLayerGetter
from .TaskNetworks import JigsawTask, GenericTask

Model_Dict ={
    'resnet': ['layer4'],
    'densenet': ['features'],
    'shufflenet': ['conv5']
}

class PIRLModel(nn.Module):
    def __init__(self, net, layer_names=None, pretrained=True, encoding_size=128, jigsaw_size=(3,3)):
        super(PIRLModel, self).__init__()
        if isinstance(net, str):
            if any([f in net for f in Model_Dict.keys()]):
                key = [f for f in Model_Dict.keys() if f in net][0]
                self.layer_names = layer_names or Model_Dict[key]
                self.net = getattr(models, net)(pretrained=pretrained)
        self.ILG = IntermediateLayerGetter(self.net, return_layers=self.layer_names)
        fake_data = torch.rand(2,3,224,224)
        _ = self.net(fake_data)
        out_size = self.ILG.output_sizes[self.layer_names[0]]
        self.GeneralRepresentation = GenericTask(out_size, encoding_size)
        self.Jigsaw = JigsawTask(out_size, encoding_size, jigsaw_size)
    
    def forward(self, image, transformed_image=None):
        _ = self.net(image)
        image_activations = self.ILG.activations[self.layer_names[0]]
        image_features = self.GeneralRepresentation(image_activations)
        
        if not transformed_image is None:
            transformed_image = torch.cat([*transformed_image], dim=0) #Collapsing batch and patch dimensions
            _ = self.net(transformed_image)
            image_activations = self.ILG.activations[self.layer_names[0]]
            transformed_image_features = self.Jigsaw(image_activations)
            return image_features, transformed_image_features
        else:
            return image_features

class PIRLLoss(nn.Module):
    def __init__(self, loss_lambda=0.9):
        super(PIRLLoss, self).__init__()
        self.loss_lambda = loss_lambda
        self.l1 = torch.nn.CrossEntropyLoss()
        self.l2 = torch.nn.CrossEntropyLoss()
    
    def forward(self, transformed_output, output, index):
        l1 = F.cross_entropy(transformed_output, index)
        l2 = F.cross_entropy(output, index)
        total = self.loss_lambda *  l1 + (1 - self.loss_lambda) * l2
        return total
        
