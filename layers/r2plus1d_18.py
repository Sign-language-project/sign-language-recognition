import torch
import torch.nn as nn
import torchvision


class R2plus1d(nn.Module):
  def __init__(self, out_dim = 226):
    super(R2plus1d, self).__init__()
    self.model = torchvision.models.video.r2plus1d_18(pretrained= True)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(500, out_dim)


  def forward(self, x):
    x = self.model(x) 
    x = self.relu(x)
    x = self.fc(X)
    return x