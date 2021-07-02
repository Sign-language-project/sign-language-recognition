
import torch
import torch.nn as nn


class Stream_a(nn.Module):
    def __init__(self, model, out_dim, trainable=False,  ckpt_path : str = None):

        super(Stream_a, self).__init__()

        self.model = model
        self.out_dim = out_dim
        self.trainable = trainable
        self.ckpt_path = ckpt_path

    def forward(self, x): # (batch, 65, 120)
        x = self.model(x[0])
        return x