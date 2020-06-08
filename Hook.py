import torch

class Hook:
    def __init__(self, layer):
        self.layer=layer
        self.layer.register_forward_hook(self.forward_fn)
        self.layer.register_backward_hook(self.backward_fn)

    def forward_fn(self, module, f_input, f_output):
        self.forward = f_output.detach().numpy()

    def backward_fn(self, module, grad_input, grad_output):
        # compute the mean over the 14x14 dimensions, to numpy
        self.backward = torch.mean(grad_output[0], dim=[2,3]).numpy()[0,:]
