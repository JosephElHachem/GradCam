import torch
import torch.nn as nn

class Hook:
    def __init__(self, model, conv2d_backcount=1):
        self.model = model
        self.conv2d_layers = []
        self.get_conv2d(self.model)
        if conv2d_backcount<len(self.conv2d_layers):
            self.hook_layer(self.conv2d_layers[-conv2d_backcount])
        else:
            raise ValueError(f'There are only {len(self.conv2d_layers)} 2D CNN layers, let 1 <= conv2d_backcount <= {len(self.conv2d_layers)}')

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
