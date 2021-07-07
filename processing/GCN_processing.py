# Importing libraries

import mediapipe as mp
import random
import torch
import cv2
import numpy as np
import os


class GCN_processing ():
    

    def make_processed_pt (self, video_pt):
        # takes Pt file of the video and return a tensor of (65,120) shape as input to stream a
  
        frame_start = 0
        frame_end = video_pt.size(0) -1
        
        frames_to_sample = self.rand_start_sampling(frame_start, frame_end) # 60 or less frame to sample from
        
        frames = []
        for i in frames_to_sample:
            frames.append(video_pt[i,:,:,:].numpy())
        xy = self.pt_to_pose(torch.tensor(frames))

        return xy # posees tensor of shape (65,120)

    def pt_to_pose(self, images, num_samples=60):
        # takes video frames and return tensor of posses of each frame 
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5) as holistic:
            poses = []
            for i in range(images.size(0)):
                results = holistic.process(cv2.cvtColor(np.array(images[i,:,:,:]), cv2.COLOR_BGR2RGB))
                xy  = self.get_pose(results)
                poses.append(xy)

            pad = None
            if len(poses) < num_samples: # make sure all videos with at least 60 frame
                num_padding = num_samples - len(poses)
                last_pose = poses[-1]
                pad = last_pose.repeat(1, num_padding)

            poses_across_time = torch.cat(poses, dim=1)
            if pad is not None:
                poses_across_time = torch.cat([poses_across_time, pad], dim=1)

        return poses_across_time

    

    def rand_start_sampling(frame_start, frame_end, num_samples=60): 
        """Randomly select a starting point and return the continuous ${num_samples} frames."""
        num_frames = frame_end - frame_start + 1

        if num_frames > num_samples:
            select_from = range(frame_start, frame_end - num_samples + 1)
            sample_start = random.choice(select_from)
            frames_to_sample = list(range(sample_start, sample_start + num_samples))
        else:
            frames_to_sample = list(range(frame_start, frame_end + 1))

        return frames_to_sample

        
    def get_pose (results):
        # takes mediabibe holistic object and return the posses of each frame in shape (65,2)
        x = [] 
        y = []
        if results.left_hand_landmarks == None:
            x.extend([0] * 21)
            y.extend([0] * 21)
        else:
            for i in range(len(results.left_hand_landmarks.landmark)):
                x.append(results.left_hand_landmarks.landmark[i].x)
                y.append(results.left_hand_landmarks.landmark[i].y)

        if results.right_hand_landmarks == None:
            x.extend([0] * 21)
            y.extend([0] * 21)
        else:
            for i in range(len(results.right_hand_landmarks.landmark)):
                x.append(results.right_hand_landmarks.landmark[i].x)
                y.append(results.right_hand_landmarks.landmark[i].y)
            
        if results.pose_landmarks == None:
            x.extend([0] * 23)
            y.extend([0] * 23)
        else:
            for i in range(23):
                x.append(results.pose_landmarks.landmark[i].x)
                y.append(results.pose_landmarks.landmark[i].y)

        
        return torch.stack([torch.tensor(x), torch.tensor(y)]).transpose_(0, 1)
        
