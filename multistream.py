import torch
from torch import nn
import torch.nn.functional as F
from layers import concate
from stream_a import Stream_a
from stream_b import Stream_b
from stream_c import Stream_c
from layers.concate import Concate
from lit_base import Base

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
      #for stream in self.streams:
      #  assert stream.out_dim == self.concate_dim , f"out_dim of stream {stream.__class__.__name__} doesn't match the concate dim"

      #make the streams lightening modules, load the checkpoints if needed
      for i in range(len(self.streams)):
        self.streams[i] = Base(self.streams[i])

        #load the checkpoints of the to-be-freezed streams
        if self.streams[i].model.trainable == False:

          assert self.streams[i].model.ckpt_path != None , f"No checkpoint path is found for {self.streams[i].__calss__.__name__}, pass the path to the class __init__"

          checkpoint = torch.load(self.streams[i].model.ckpt_path)
          self.streams[i].load_state_dict(checkpoint['state_dict'])

          for j, param in enumerate(self.streams[i].parameters()):
            param.requires_grad = False
            
        self.streams[i].model.model.fc = nn.Linear(400, self.concate_dim)


      #model blocks

      # The list of the streams to be trained on
      self.streams_list = nn.ModuleList(self.streams)

      # The layer to concate and give weights to the streams outputs
      self.concate = Concate(n_streams= len(streams), in_dim= self.concate_dim, out_dim= self.latent_dim )

      #the classification layers
      self.classification = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(self.dropout),
        nn.Linear(self.latent_dim, self.num_classes)
      )

  def forward(self, x):

    #run the streams
    streams_outs = []
    for index, stream in enumerate(self.streams_list):
      output = stream(x[index])
      streams_outs.append(output)

    #concate and average
    concated = self.concate(streams_outs)

    #classification
    out = self.classification(concated)

    return out
