class IntermediateLayerGetter(object):
    """
    Saves intermediate layers in a dictionary
    """
    def __init__(self, net, return_layers=[]):
        self.activations = {}
        self.output_sizes = {}
        for name, module in net.named_modules():
            if name in return_layers:
                module.register_forward_hook(self.get_hook(name))

    def get_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = output
            self.output_sizes[name] = output.size(1)
        return hook