import torch
from torch import nn
import torch.nn.functional as F
from stream_b import Stream_b
from stream_c import Stream_c
from layers.concate import Concate


class Stream_bc (nn.Module):

    def __init__(self ,models , trainables, concat_dim, num_classes, device, raft_parameters_path, raft_iters = 12):
      """
      a model for training the streams b and c together.

      args:
      ==================================
        models (list[str]) : a list that contain the chosen model for every stream. (models[0] >> stream b , models[1] >> stream c)
        trainables (list[bool]) : to choose whether to train the pretrained models of b and c or not.
        concat_dim : the output dim of the pretrained models. the dim at which the output of the streams will be concated together.
        num_classes : the number of classes of the main dataset.
        device : the device at which to train the model.
        raft_parameters_path : the path of the pretrained raft parameters.
        raft_iters : number of iterations of the raft.
      """

      super().__init__()

      self.stream_b_model = Stream_b(models[0], concat_dim, raft_parameters_path, device, raft_iters , trainables[0])

      self.stream_c_model = Stream_c(models[1], concat_dim, trainables[1])

      self.concat = Concate(n_streams = 2, in_dim = concat_dim, out_dim = concat_dim)

      self.linear = nn.Linear(concat_dim, num_classes)


    def forward(self, x):
      #X >> tuple of two (videos , videos_raft)

      videos = x[0]         #(B, C, T, H, W)
      videos_raft = x [1]   #(B, C, T, H, W)

      stream_b_out = self.stream_b_model(videos_raft) #(b, concat_dim)

      stream_c_out = self.stream_c_model(videos) #(b, concat_dim)

      out = self.concat([stream_b_out, stream_c_out]) #(b  , concat_dim)

      out = self.linear(out) #(b  , num_classes)

      return out
