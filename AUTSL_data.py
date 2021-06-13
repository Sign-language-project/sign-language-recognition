import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import os
import pathlib
from transforms import RandomHorizontalFlip, Transform, Transform_raft
import numpy as np


# Dataset Module
class TSL_dataset(Dataset):
  """
  This class will be a general Dataset class for the WLASL data training.
  you can specify the set you need (100, 300, 1000, 2000) and the subset you will use (train, val, test)
  using the class attributes.
  the class assumes that the ZewZew dir is at "MyDrive", so you will have no paths errors

  args:
    set: str or int, the set you want to train or test on. must be in [100, 300, 1000, 2000]. default = '100'
    subset: str, the subset of the dataset object. must be in ['train', 'val', 'test']. default = 'train'
    max_frames: int, number of max frames in the video. default = 12, the mean
    streams: list, list of strings of the used streams 'a', 'b', 'c'.
  """
  def __init__(self,
               subset = 'train',
               videos_path_root = "/content/drive/MyDrive/ZewZew/AUTSL",
               train_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/train_labels.csv",
               val_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/val_labels.csv",
               test_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/test_labels.csv",
               img_size = 256,
               max_frames = 16,
               streams = None,
               make_raft_transform = False,
               ):
    super(TSL_dataset).__init__()
    self.subset = subset
    assert subset in ['train', 'val', 'test'], 'subset must be in ["train", "val", "test"]'

    self.img_size = img_size
    self.max_frames = max_frames
    if subset == 'train':
      self.csv_path = train_csv_file
    elif subset == 'val':
      self.csv_path = val_csv_file
    elif subset == 'test':
      self.csv_path = test_csv_file
    #get the csv fila as a DataFrame
    self.df = self.get_df()
    self.video_paths = self.df['video_name'].tolist()
    self.labels = self.df['class'].tolist()
    self.path_root = videos_path_root

    #video tansforms
    self.transforms = Transform(size = img_size)

    #Raft tansforms
    self.transforms_raft = Transform_raft(size = img_size)

    self.train_transforms = torchvision.transforms.Compose([
                                       RandomHorizontalFlip(),
    ])


  def get_df(self):
    """
    method to get the df to be used.
    """
    df = pd.read_csv(self.csv_path) #read the file

    #drop any video with 0 frames
    df = df[df['num_frames'] != 0].reset_index()
    return df

  def __len__(self):
    return len(self.df)

  @staticmethod
  def get_maxframes(self,frames):
    """
    make each video in the maximum number of frames
    params:
      frames: np array of one video frames with shape (T, H, W, C)

    returns:
      frames: nd array (T, H, W, C)
    """
    #if no. of frames is greater than the max_frmes I will drop randomly some of them
    if frames.shape[0] > self.max_frames:
      num_to_drop = frames.shape[0] - self.max_frames   #no. of frames needed to be droped
      to_drop = np.random.choice(np.arange(frames.shape[0]), num_to_drop, replace= False )  #choose rondomly frames to drop
      frames = np.delete(frames, to_drop, axis= 0)  #drop

    #if no. of frames is less than the max_frames I will pad with zeros frames
    elif frames.shape[0] < self.max_frames:
      num_to_append = self.max_frames - frames.shape[0]  #no of padding frames needed
      pad_frames = np.zeros((num_to_append, *frames.shape[1:]))  #creat the padding frames with the shape (num_to_append, c, h, w)
      frames = np.append(frames, pad_frames , axis= 0)   #pad

    return frames

  def __getitem__(self, index):
    """
    """
    video_path = os.path.join(self.path_root,self.video_paths[index])
    label = self.labels[index]
    video = torch.load(video_path) #shape (C , T, H, W)
    video = video.permute(1, 0, 2, 3) #shape (T , C, H, W)

    video = self.get_maxframes(self, video.numpy())  #make the frames num equal the max_frames

    #if self.subset == 'train':
    #  video = self.train_transforms(video)


    video = video.transpose(0, 2, 3, 1) #shape (T , H, W, C)

    if make_raft_transform:
      video_raft = self.transforms_raft(video) #shape (C ,T, H, W)

    if self.transforms:
      video = self.transforms(video) #shape (C ,T, H, W)

    if make_raft_transform:
      video = (video , video_raft) #tuple the holds the preprocessed data for the normal 3D conv and raft optical flow

    #zip the sample
    sample = (video, label)

    return sample
