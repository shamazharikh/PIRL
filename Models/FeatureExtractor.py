import torch.nn as nn
class IntermediateLayerGetter(object):
    """
    Saves intermediate layers in a dictionary
    """
    def __init__(self, net, return_layers=[]):
        # super(IntermediateLayerGetter, self).__init__()
        self.activations = {}
        self.output_sizes = {}
        for name, module in net.named_modules():
            if name in return_layers:
                module.register_forward_hook(self.get_hook(name))

    def get_hook(self, name):
        def hook(module, input, output):
            device = output.get_device()
            self.activations.setdefault(name, {})[device] = output   
            self.output_sizes[name] = output.size(1)
        return hook
