import torch
from torch import nn
import torch.nn.functional as F
from layers import concate
from stream_a import stream_a
from stream_b import Stream_b
from stream_c import Stream_c
from layers.concate import Concate


class MultiStream(nn.Module):
  """
  args:
    streams: list, list of strings of the used streams 'a
    trainables: list, list of bools, True for trainable, BE SURE TO PASS IT WITH THE SAME ORDER AS strems ARG. if None will train all models with the arg `model` backbone.
    b_backbone: str, the 3D backbone model name of stream b, if you passed it don't pass the arg "model" default None wil. if None will train all models with the arg `model` backbone.
    c_backbone: str, the 3D backbone model name of stream c, if you passed it don't pass the arg "model".
  """

  def __init__(self,
              streams: list,
              concate_dim: int = 512,
              num_classes: int = 226,
              latent_dim: int = 512,
              dropout = 0.4
  ):
              
      super(MultiStream, self).__init__()
      self.concate_dim = concate_dim
      self.num_classes = num_classes
      self.latent_dim = latent_dim
      self.dropout = dropout
      #Get streams
      self.streams = streams

      #make sure that the output of the each stream match with the input of the concate layer
      for stream in self.streams:
        assert stream.out_dim == self.concate_dim , f"out dim of stream {stream.__class__.__name__} dont match with the concate dim"
      
      self.concate = Concate(n_streams= len(streams), in_dim= self.concate_dim, out_dim= self.latent_dim )
      self.classification = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.latent_dim, self.num_classes)
      )

  def forward(self, x):

    #run the streams
    streams_outs = []
    for stream in self.streams:
      output = stream(x)
      streams_outs.append(output)
    
    #concate and average
    concated = concate(streams_outs)

    #classification
    out = self.classification(concated)

    return out




"""
class Stream_bc (nn.Module):

    def __init__(self ,models , trainables, concat_dim, num_classes, device, raft_parameters_path, raft_iters = 12):
      
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
"""