from pytorchvideo.models.hub import efficient_x3d_xs
import torch.nn as nn
import torch


class E_x3d_xs(nn.Module):
    def __init__(self, out_dim = 226 ):
        super(E_x3d_xs, self).__init__()
        self.x3d = efficient_x3d_xs(pretrained= True, progress= True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(400, out_dim)
    
    def forward(self, x):
        x = self.x3d()
        x = self.relu(x)
        x = self.fc(x)

        return x
    