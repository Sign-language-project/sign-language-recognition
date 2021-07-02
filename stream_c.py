import torch
import torch.nn as nn
from layers import efficient_x3d_xs, r2plus1d_18, r2plus1d_50


class Stream_c(nn.Module):
    """
    Args:
        model: str, a strig with the right name of the needed model, now supported "x3d", "r2plus1d"
    """
    def __init__(self, model, out_dim, trainable = True, ckpt_path: str = None):
        super(Stream_c, self).__init__()

        assert model in ['x3d', 'r2plus1d','r2plus1d_50'] , "models suported for stream C are 'x3d', 'r2plus1d', 'r2plus1d_50'"

        self.out_dim = out_dim
        self.trainable = trainable
        self.ckpt_path = ckpt_path

        if model == 'x3d':
            self.model = efficient_x3d_xs.E_x3d_xs(out_dim)
        elif model == 'r2plus1d':
            self.model = r2plus1d_18.R2plus1d(out_dim)
        elif model == 'r2plus1d_50':
            self.model = r2plus1d_50.R2plus1d_50(out_dim)

    def forward(self, x):
        x = self.model(x)
        return x
