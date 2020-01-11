import torch.nn as nn
class IntermediateLayerGetter(nn.Module):
    """
    Saves intermediate layers in a dictionary
    """
    def __init__(self, net, return_layers=[]):
        super(IntermediateLayerGetter, self).__init__()
        self.activations = {}
        self.output_sizes = {}
        self.num_calls = 0
        for name, module in net.named_modules():
            if name in return_layers:
                module.register_forward_hook(self.get_hook(name))

    def get_hook(self, name):
        def hook(module, input, output):
            print("inside the hook ......", self.num_calls, "times", output.get_device())
            self.num_calls += 1
            self.activations[name] = output
            self.output_sizes[name] = output.size(1)
        return hook
