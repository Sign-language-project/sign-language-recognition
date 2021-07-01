
import torch
import torch.nn as nn


class Stream_a(nn.Module):
    def __init__(self, model, out_dim, trainable=False,  ckpt_path : str = None):

        super(Stream_a, self).__init__()

        self.model = model
        self.out_dim = out_dim

        if not trainable:
            #self.model.fc = nn.Identity()
            #check the path of the checkpoint
            assert ckpt_path != None , "No checkpoint path is found, pass the path to the class __init__"
            
            #load the checkpoint
            checkpoint = torch.load(ckpt_path)
            weights = checkpoint['state_dict']
            new_weights = {}
            for key in weights.keys():
                new_weights[key[6:]] = weights[key]
            self.model.model.load_state_dict(new_weights)
            
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
            self.model.fc = nn.Linear(400, out_dim)

    def forward(self, x): # (batch, 65, 120)
        x = self.model(x[0])
        return x
