import torch
from PIL import Image
import functools
import numpy as np
import librosa

# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import torch.utils.data as data


def video_loader(video_dir_path):
    # Load video data from numpy file
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        # Convert each frame to PIL Image
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

def get_default_video_loader():
    # Return a partial function that uses video_loader as the default video loader
    return functools.partial(video_loader)

def load_audio(audiofile, sr):
    # Load audio data using librosa
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr

def get_mfccs(y, sr):
    # Compute MFCC features using librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

def make_dataset(subset, annotation_path):
    # Read annotation file and create dataset
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
        
    dataset = []
    for line in annots:
        filename, audiofilename, label, trainvaltest = line.split(';')        
        if trainvaltest.rstrip() != subset:
            continue
        
        sample = {'video_path': filename,                       
                  'audio_path': audiofilename, 
                  'label': int(label)-1}
        dataset.append(sample)
    return dataset 
       

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type = 'audiovisual', audio_transform=None):
        # Initialize RAVDESS dataset
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform=audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    def __getitem__(self, index):
        # Get item at the given index
        target = self.data[index]['label']
                

        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            # Process video data
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                # Apply spatial transformation to each frame
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                # Return video clip and target label
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            # Process audio data
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 # Apply audio transformation
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if self.data_type == 'audio':
                # Return audio features and target label
                return audio_features, target
        if self.data_type == 'audiovisual':
            # Return audio features, video clip, and target label
            return audio_features, clip, target  

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)
