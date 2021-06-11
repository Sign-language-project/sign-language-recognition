
import torch
import torch.nn as nn


class Concate(nn.Module):
  def __init__(self, n_streams = 1, in_dim = 512, out_dim = 512):
    super(Concate, self).__init__()
    
    self.n_streams = n_streams
    self.fc = nn.Linear(self.n_streams * in_dim , out_dim )

  def forward(self, x: list):
    """
    x -> list of streams outputs of 3d conv with shapes = [bs, ts, in_dim]
    """
    assert len(x) == self.n_streams, "number of streams does not match with the list length"

    x = torch.stack(x, dim = 2) #[bs, ts, n_streams, in_dim]
    bs, ts, _, _ = x.shape
    x = x.reshape(bs, ts, -1)
    x = self.fc(x)  # [bs, ts, out_dim]
    return x