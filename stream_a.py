
import torch
import torch.nn as nn

from layers.GC_block import GraphConvolution_att, GC_Block



class stream_a(nn.Module):
    def __init__(self, num_class, input_feature=100, hidden_feature=100, p_dropout=0.3, num_stage=20, is_resi=True):
        super(stream_a, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution_att(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(65 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        # self.gc7 = GraphConvolution_att(hidden_feature, output_feature)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        # self.fc1 = nn.Linear(55 * output_feature, fc1_out)
        self.fc_out = nn.Linear(hidden_feature, num_class)
        #self.fc_out = nn.Linear(65 * hidden_feature, num_class)

    def forward(self, x): # (batch, 55, 100)
        y = self.gc1(x[0]) # (batch, 55, 100)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y) # (batch, 55, 100)

        out = torch.mean(y, dim=1)  # (batch, 65, 100) --> # (batch, 100)
        #out = y.view(b, -1)

        out = self.fc_out(out)

        return out # (batch, 100)
