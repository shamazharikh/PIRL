import torch
import torch.nn as nn
import torchvision.models as models

from FeatureExtractor import IntermediateLayerGetter
from TaskNetworks import JigsawTask, GenericTask
Model_Dict ={
    'resnet': 'layer4',
    'densenet': 'features'
}
class PIRLModel(nn.Module):
    def __init__(self, net, layer_names=None, encoding_size=128, jigsaw_size=(3,3)):
        if isinstance(net, str):
            if any([f in net for f in Model_Dict.keys()]):
                key = [f for f in Model_Dict.keys() if f in net][0]
                self.layer_names = layer_names or Model_Dict[key]
                self.net =getattr(models, net)(pretained=True)
        self.ILG = IntermediateLayerGetter(self.net, return_layers=self.layer_names)
        fake_data = torch.rand(2,3,224,224)
        _ = net(fake_data)
        out_size = ILG.output_sizes[layer_names[0]]
        self.GeneralRepresentation = GenericTask(out_size, encoding_size)
        self.Jigsaw = JigsawTask(out_size, encoding_size, jigsaw_size)
    
    def forward(self, image, transformed_image):
        _ = self.net(image)
        image_activations = self.ILG.activations[self.layer_names[0]]
        image_features = self.GeneralRepresentation(image_activations)
        
        _ = self.net(transformed_image)
        image_activations = self.ILG.activations[self.layer_names[0]]
        transformed_image_features = self.Jigsaw(image_activations)
        
        return image_features, transformed_image_features


