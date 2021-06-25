import torch.nn as nn
from pytorchvideo.models.r2plus1d import create_r2plus1d
from torch.hub import load_state_dict_from_url


class R2plus1d_50(nn.Module):
  def __init__(self, out_dim = 226):

    super(R2plus1d_50, self).__init__()

    self.model = create_r2plus1d( head_activation = None, dropout_rate=0.5)

    path = "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/R2PLUS1D_16x4_R50.pyth"
    checkpoint = load_state_dict_from_url( path, progress=True, map_location="cpu")
    state_dict = checkpoint["model_state"]
    self.model.load_state_dict(state_dict)

    self.relu = nn.ReLU()
    self.fc = nn.Linear(400, out_dim)


  def forward(self, x):
    x = self.model(x)
    x = self.relu(x)
    x = self.fc(x)
    return x
