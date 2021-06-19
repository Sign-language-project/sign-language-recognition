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
    streams: list, list of strings of the used streams 'a', 'b', 'c'. if none, default is stream c
  """
  def __init__(self,
               streams = 'c',
               subset = 'train',
               videos_path_roots = ["/content/drive/MyDrive/ZewZew/AUTSL"],
               train_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/train_labels.csv",
               val_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/val_labels.csv",
               test_csv_file = "/content/drive/MyDrive/ZewZew/AUTSL/processed_csv_7fps/test_labels.csv",
               img_size = 256,
               max_frames = 16,

               ):
    super(TSL_dataset).__init__()
    self.subset = subset
    self.streams = streams
    if type(self.streams) != 'str':
      self.streams = "".join(self.streams)
    
    assert subset in ['train', 'val', 'test'], 'subset must be in ["train", "val", "test"]'

    if self.streams:
        assert sum( [ (stream_letter.lower() in ['a', 'b','c']) for stream_letter in streams] ) == len(self.streams) , "you have entered a non valid stream letter, letters must be in 'abc'"

    self.path_root_a = videos_path_roots[ self.streams.find('a') ]
    self.path_root_b = videos_path_roots[ self.streams.find('b') ]
    self.path_root_c = videos_path_roots[ self.streams.find('c') ]

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
    #df = df[df['num_frames'] != 0].reset_index()
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
  
  def get_sampled_video(self, video, org_fps):
    
    #Get the sampled frames idxs
    sampled_frames_idx = np.linspace(0, video.shape[0]-1, self.max_frames)
    #new video tensor
    sampled_video = video[sampled_frames_idx, :,:,:]
    return sampled_video

  def prepare_video(self, path):
    """
    upload the video, prepare it for the training
    Use this method if want to upload row video
    """

    #load the video
    video, _, _ = torchvision.io.read_video(path, pts_unit= 'sec')

    #sample the video
    video = self.get_sampled_video(video, org_fps = 30)
    video = video.permute(3, 0, 1, 2) #T, C, H, W
    return video

  def __getitem__(self, index):
    """
    """

    label = self.labels[index]

    video_b = None

    video = []

    if 'a' in self.streams:
      path = f'{self.video_paths[index][:-4]}.pt'
      pose_path = os.path.join(self.path_root_a ,path)

      pose = torch.load(pose_path) #shape (65 , 120)

      video.append(pose)


    if 'b' in self.streams:

        video_b_path = os.path.join(self.path_root_b,self.video_paths[index])

        #video_b = self.prepare_video(video_b_path) #shape (C , T, H, W)
        video_b = torch.load(video_b_path)

        video_b = video_b.permute(1, 0, 2, 3) #shape (T , C, H, W)

        video_b = self.get_maxframes(self, video_b.numpy())  #make the frames num equal the max_frames

        video_b = video_b.transpose(0, 2, 3, 1) #shape (T , H, W, C)

        video_raft = self.transforms_raft(video_b) #shape (C ,T, H, W)

        video.append(video_raft)


    if  self.streams == None or 'c' in self.streams:

        if (video_b is not None) and self.transforms:

            video_c = self.transforms(video_b) #shape (C ,T, H, W)

            video.append(video_c)

        elif (video_b is not None):

            video.append(video_b)


        else:

            video_c_path = os.path.join(self.path_root_c,self.video_paths[index])

            #video_c = self.prepare_video(video_c_path)
            video_c = torch.load(video_c_path)

            video_c = video_c.permute(1, 0, 2, 3) #shape (T , C, H, W)

            video_c = self.get_maxframes(self, video_c.numpy())  #make the frames num equal the max_frames

            video_c = video_c.transpose(0, 2, 3, 1) #shape (T , H, W, C)

            if self.transforms:

                video_c = self.transforms(video_c) #shape (C ,T, H, W)

            video.append(video_c)


    sample = (*video, label)

    return sample
