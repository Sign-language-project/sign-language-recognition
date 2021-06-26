#importing the libraries we need
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import nn
from layers.RAFT.raft import RAFT
import argparse
from layers import efficient_x3d_xs, r2plus1d_18, r2plus1d_50



class Stream_b (nn.Module):

    def __init__(self, model, out_dim, raft_parameters_path, device, raft_iters = 12 , trainable = True, ckpt_path: str = None):

        super().__init__()

        assert model in ['x3d', 'r2plus1d','r2plus1d_50'] , "models suported for stream B are 'x3d', 'r2plus1d', 'r2plus1d_50'"

        args = self.get_args() #get the args which are the input parameters for the model.
        self.device = device
        self.out_dim = out_dim
        self.raft_iters = raft_iters

        self.raft_model = RAFT(args[0])
        self.raft_model = torch.nn.DataParallel(self.raft_model)
        self.raft_model.load_state_dict(torch.load(raft_parameters_path, map_location = device))
        self.raft_model = self.raft_model.module.to(self.device)

        for param in self.raft_model.parameters():
            param.requires_grad = False

        self.batch = nn.BatchNorm3d(2)
        self.conv3di = nn.Conv3d(2 , 3 , (3,3,3) , padding = 1)
        self.batch = nn.BatchNorm3d(3)

        if model == 'x3d':
            self.model = efficient_x3d_xs.E_x3d_xs(out_dim)
        elif model == 'r2plus1d':
            self.model = r2plus1d_18.R2plus1d(out_dim)
        elif model == 'r2plus1d_50':
            self.model = r2plus1d_50.R2plus1d_50(out_dim)

        if not trainable:
            self.model.fc = nn.Identity()
            #check the path of the checkpoint
            assert ckpt_path != None , "No checkpoint path is found, pass the path to the class __init__"

            #load the checkpoint
            checkpoint = torch.load(ckpt_path)
            model.model.load_state_dict(checkpoint['model'])

            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False
            self.model.fc = nn.Linear(400, out_dim)



    def forward(self , images_batch):

        #images should be on the shape of B x C x T x H x W . H and W must be dividable by 8

        images_batch = images_batch.permute(0,2,1,3,4).to(self.device) #shape  B x T x C x H x W


        raftout = []

        for images in images_batch:

            self.raft_model.eval()

            with torch.no_grad():

                _, raft_out = self.raft_model(images[:-1], images[1:], iters=self.raft_iters, test_mode=True)
                raftout.append(raft_out)

        raftout = torch.stack(raftout) #B , T-1 , C , H , W >>>> c = 2
        raftout = raftout.permute(0,2,1,3,4).to(self.device) #shape  B x C x T-1 x H x W >>>> c = 2
        #print(raftout.shape)
        out = self.conv3di(raftout) #shape  B x C x T-1 x H x W >>>>> C = 3
        #print(out.shape)
        out = self.model(out) #shape  B x 512
        #print(out.shape)
        return out


    @staticmethod
    def get_args():

        parser = argparse.ArgumentParser()
        parser.add_argument('--name', default='raft', help="name your experiment")

        parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])

        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--small', action='store_true', help='use small model')
        parser.add_argument('--iters', type=int, default=12)
        parser.add_argument('--wdecay', type=float, default=.00005)
        parser.add_argument('--epsilon', type=float, default=1e-8)
        parser.add_argument('--clip', type=float, default=1.0)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
        parser.add_argument('--add_noise', action='store_true')
        args = parser.parse_known_args()

        return args
