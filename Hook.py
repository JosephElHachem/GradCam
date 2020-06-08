import torch
import torch.nn as nn

class Hook():
    def __init__(self, model):
        self.model = model
        self.conv2d_layers = []
        self.get_conv2d(self.model)
        self.hook_layer(self.conv2d_layers[-1])

    def get_conv2d(self, model):
        # saving all conv2d layers in self.conv2d_layers
        for layer in model.children():
            if isinstance(layer, nn.Sequential):
                self.get_conv2d(layer)
            else:
                if isinstance(layer, nn.Conv2d):
                    self.conv2d_layers.append(layer)

    def hook_layer(self, layer):
        # hooking layer
        self.layer=layer
        self.layer.register_forward_hook(self.forward_fn)
        self.layer.register_backward_hook(self.backward_fn)

    def forward_fn(self, module, f_input, f_output):
        self.forward = f_output.detach().numpy()

    def backward_fn(self, module, grad_input, grad_output):
        # compute the mean over the 14x14 dimensions, to numpy
        self.backward = torch.mean(grad_output[0], dim=[2,3]).numpy()[0,:]
