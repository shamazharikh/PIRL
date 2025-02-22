import torch
from torch.autograd import Function
from torch import nn
import math

class LinearAverageOp(Function):
    @staticmethod
    def forward(self, features, transformed_features, indices, memory, params):
        T = params[0].item()

        # inner product
        out_features = torch.mm(features.data, memory.t())
        out_features.div_(T) # batchSize * N
        
        # inner product
        out_trans_features = torch.mm(transformed_features.data, memory.t())
        out_trans_features.div_(T) # batchSize * N
        
        # inner product for input similarity
        out_similarity = (features.data * transformed_features.data).sum(dim=-1, keepdim=True)
        out_trans_features.div_(T) # batchSize 
        self.save_for_backward(features, transformed_features, memory, indices, params)

        return out_trans_features, out_features, out_similarity

    @staticmethod
    def backward(self, grad_trans_output, grad_output, grad_output_sim):
        features, transformed_features, memory, indices, params = self.saved_tensors
        T = params[0].item()
        momentum = params[1].item()
        
        # add temperature
        grad_trans_output.data.div_(T)
        grad_output.data.div_(T)

        # gradient of linear
        grad_output = torch.mm(grad_output.data, memory)
        grad_output.resize_as_(features)

        grad_trans_output = torch.mm(grad_trans_output, memory)
        grad_trans_output.resize_as_(transformed_features)
        
        # update the non-parametric data
        weight_pos = memory.index_select(0, indices.data.view(-1)).resize_as_(features)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(features.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, indices, updated_weight)
        
        return grad_output, grad_trans_output, None, None, None

class LinearAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.07, momentum=0.5):
        super(LinearAverage, self).__init__()
        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))

    def forward(self, image_features, transformed_image_features, indices):
        out = LinearAverageOp.apply(
            image_features, 
            transformed_image_features,
            indices,
            self.memory,
            self.params)
        return out

