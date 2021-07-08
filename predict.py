import torch
import pandas as pd
from processing.video_processing import ProcessVideo
from processing.GCN_processing import GCN_processing
from transforms import Transform, Transform_raft
from pytorch_lightning import LightningModule


class predict:
    def __init__(
        self,
        model: LightningModule,
        streams: str,
        words_Ids_csv_path = "words_ids\SignList_ClassId_TR_EN.csv",
        device = 'cpu'
        ):

        self.model = model
        self.device = device
        self.word_id_df = pd.read_csv(words_Ids_csv_path)
        self.transforms = Transform()
        self.transforms_raft = Transform_raft()
        self.streams = streams

    def prepare_for_stream_a(self, video_path):

        video = ProcessVideo(sample= False)(video_path) #(T, C, H, W)

        video = video.permute(0, 3, 1, 2) #(T, H, W, C)

        processed = GCN_processing()(video)

        return processed
    
    def prepare_for_stream_c(self, video_path):

        video = ProcessVideo()(video_path) #(T, C, H, W)

        video = video.permute(0, 2, 3, 1)  #(T , H, W, C)

        processed = self.transforms(video)

        return processed
    
    def  prepare_for_stream_b(self, video_path):

        video = ProcessVideo()(video_path) #(T, C, H, W)

        video = video.permute(0, 2, 3, 1) #(T, H, W, C)

        processed = self.transforms_raft(video) #shape (C ,T, H, W)

        return processed
    
    def __call__(self, video_path):
        
        videos = []

        #Get the input model
        for stream in self.streams:
            if stream == 'a':
                video = self.prepare_for_stream_a(video_path)
                videos.append(video)
            if stream == 'b':
                video = self.prepare_for_stream_b(video_path)
                videos.append(video)
            if stream == 'c':
                video = self.prepare_for_stream_c(video_path)
                videos.append(video)

        #Eval mode for the model        
        model = self.model.to(self.device)
        model.eval()

        #predict
        out = model(*videos)
        out = out.to('cpu')
        soft_out = torch.softmax(out, dim =-1)

        #get the outputs
        top_5 = torch.topk(soft_out, k =5, sorted= True)
        words_indices = top_5[1].numpy()
        words_probs = top_5[0].numpy()

        words = self.word_id_df.loc[words_indices, 'EN'].tolist()  #English words

        return {'IDs': words_indices,
                'Words': words,
                'Probs': words_probs }

            

    



    