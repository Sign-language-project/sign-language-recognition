import torch
from torch import nn
import torch.nn.functional as F
from layers import concate
from stream_a import Stream_a
from stream_b import Stream_b
from stream_c import Stream_c
from layers.concate import Concate


class MultiStream(nn.Module):
  """
  args:
    streams: list, list of the models objects. Make sure that the output dim of each the same and equals the concate dim
    concate_dim: int, the input dim of the concate layer.
    num_classes: int, the number of classes.
    latent_dim: int, the output dim of the concate layer.
    dropout, float, probability of an element to be zeroed in the classification layer. Default: 0.5
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
      self.streams = streams

      #make sure that the output of the each stream match with the input of the concate layer
      for stream in self.streams:
        assert stream.out_dim == self.concate_dim , f"out_dim of stream {stream.__class__.__name__} doesn't match the concate dim"

      self.concate = Concate(n_streams= len(streams), in_dim= self.concate_dim, out_dim= self.latent_dim )

      self.classification = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.latent_dim, self.num_classes)
      )

  def forward(self, *x):

    #run the streams
    streams_outs = []
    for index, stream in enumerate(self.streams):
      output = stream(x[index])
      streams_outs.append(output)

    #concate and average
    concated = self.concate(streams_outs)

    #classification
    out = self.classification(concated)

    return out
