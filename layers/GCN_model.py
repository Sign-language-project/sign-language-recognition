
import torch
import torch.nn as nn

from layers.GC_block import GraphConvolution_att, GC_Block



class Stream_a_model(nn.Module):
    def __init__(self, out_dim, input_feature=120,
                 hidden_feature=100, p_dropout=0.3,
                 num_stage=20, is_resi=True):


        super(Stream_a_model, self).__init__()
        self.num_stage = num_stage
        self.out_dim = out_dim
        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(65 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc = nn.Linear(hidden_feature, out_dim)

    def forward(self, x): # (batch, 65, 120)
        y = self.gc1(x) # (batch, 65, 120)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y) # (batch, 65, 120)

        out = torch.mean(y, dim=1)  # (batch, 65, 100) --> # (batch, 120)

        out = self.fc(out)

        return out # (batch, 100)
