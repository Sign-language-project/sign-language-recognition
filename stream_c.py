import torch
import torch.nn as nn
from layers import efficient_x3d_xs, r2plus1d_18


class Stream_c(nn.Module):
    """
    Args:
        model: str, a strig with the right name of the needed model, now supported "x3d", "r2plus1d" 
    """
    def __init__(self, model, out_dim, trainable = True):
        super(Stream_c, self).__init__()
        
        assert model in ['x3d', 'r2plus1d'] , "models suported for stream A are 'x3d', 'r2plus1d'"

        if model == 'x3d':
            self.model = efficient_x3d_xs.E_x3d_xs(out_dim)
        elif model == 'r2plus1d':
            self.model = r2plus1d_18.R2plus1d(out_dim)

        if not trainable:
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
            self.model.fc.requires_grad = True
    

    def forward(self, x):
        x = self.model(x)
        return x

