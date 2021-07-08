import torch 
import torchvision
import numpy as np



class ProcessVideo:
    def __init__(self,sample: True, num_frames = 16):
        self.num_frames = num_frames
        self.sample = sample   

    def prepare_video(self, path):
        """
        upload the video, prepare it for the training
        Use this method if want to upload row video
        """

        #load the video
        video, _, _ = torchvision.io.read_video(path, pts_unit= 'sec')

        #sample the video
        if self.sample:
            video = self.get_sampled_video(video, org_fps = 30)
        
        #print(video.shape)
        #video = video.permute(3, 0, 1, 2) #T, C, H, W
        return video

    
    def get_sampled_video(self, video, org_fps):
        
        #Get the sampled frames idxs
        sampled_frames_idx = np.linspace(0, video.shape[0]-1, self.num_frames)
        #new video tensor
        sampled_video = video[sampled_frames_idx, :,:,:]
        return sampled_video


    def __call__(self, video_path):

        #Get and Process the video
        video_tensor = self.prepare_video(video_path)

        return video_tensor
