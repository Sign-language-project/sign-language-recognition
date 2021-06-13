import torch
from torch import nn
import torch.nn.functional as F

class Concate(nn.Module):

    def __init__(self, n_streams = 2, in_dim = 512, out_dim = 512):
        super(Concate, self).__init__()

        self.n_streams = n_streams
        
        self.alpha = torch.nn.Parameter(torch.zeros(in_dim , n_streams)) #512 , 2

        self.fc = nn.Linear(in_dim, out_dim )
    
    def forward(self, x: list):
        """
        x -> list of streams outputs of 3d conv with shapes = [bs, in_dim]
        """
        assert len(x) == self.n_streams, "number of streams does not match with the list length"

        x = torch.stack(x, dim = -1) #[bs, in_dim, n_streams]
        
        x = torch.mul( x , torch.sigmoid(self.alpha) ).sum(-1) #[bs, in_dim, n_streams]
        
        x = self.fc(x)  # [bs, out_dim]
        

        return x