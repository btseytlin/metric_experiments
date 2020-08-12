import torch
from torch import nn
import torchvision
from pytorch_metric_learning.utils import common_functions

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=True))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

def get_trunk(device):
    # Set trunk model and replace the softmax layer with an identity function
    trunk = torchvision.models.resnet18(pretrained=True)
    trunk_output_size = trunk.fc.in_features
    trunk.fc = common_functions.Identity()
    trunk = torch.nn.DataParallel(trunk.to(device))
    return trunk, trunk_output_size

def get_embedder(trunk_output_size, layers, device):
    # Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
    embedder = torch.nn.DataParallel(MLP([trunk_output_size, *layers]).to(device))
    return embedder