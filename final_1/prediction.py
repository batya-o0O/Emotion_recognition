import torch
from PIL import Image
import functools
import numpy as np
import librosa
from torch import nn
from models.multimodalcnn import MultiModalCNN
from opts import opts
from transforms import Compose, ToTensor

import torch.utils.data as data

# Function to load video frames from a directory
def video_loader(video_dir_path):
    video = np.load(video_dir_path)    
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i,:,:,:]))    
    return video_data

# Function to get the default video loader
def get_default_video_loader():
    return functools.partial(video_loader)

# Function to load audio from a file
def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr=sr)
    y = audios[0]
    return y, sr

# Function to extract MFCC features from audio
def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc

# Function to create the dataset
def make_dataset(subset, annotation_path):
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

# Custom dataset class for RAVDESS dataset
class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type='audiovisual', audio_transform=None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type 

    def __getitem__(self, index):
        target = self.data[index]['label']
        
        if self.data_type == 'video' or self.data_type == 'audiovisual':        
            path = self.data[index]['video_path']
            clip = self.loader(path)
            
            if self.spatial_transform is not None:               
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]            
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3) 
            
            if self.data_type == 'video':
                return clip, target
            
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            y, sr = load_audio(path, sr=22050) 
            
            if self.audio_transform is not None:
                 self.audio_transform.randomize_parameters()
                 y = self.audio_transform(y)     
                 
            mfcc = get_mfccs(y, sr)            
            audio_features = mfcc 

            if self.data_type == 'audio':
                return audio_features, target
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  
        
    def __len__(self):
        return len(self.data)

# Function to get the test dataset
def get_test_set(opt, spatial_transform=None, audio_transform=None):
    subset = 'testing'
    
    test_data = RAVDESS(
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return test_data

# Function to get the validation dataset
def get_val_set(opt, spatial_transform=None, audio_transform=None):
    subset = 'validation'
    
    test_data = RAVDESS(
        opt.annotation_path,
        subset,
        spatial_transform=spatial_transform, data_type='audiovisual',audio_transform=audio_transform)
    return test_data

# Function to compute softmax values for each set of scores
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Function for prediction
def prediction(lateFusion=False, test=True):
    opt = opts

    video_transform = Compose([
                    ToTensor(255)])
    
    if test:
        test_data = get_test_set(opt, spatial_transform=video_transform) 
    else:
        test_data = get_val_set(opt, spatial_transform=video_transform) 

    model = MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)
    model = model.to(opt.device)
    model = nn.DataParallel(model, device_ids=None)
    best_state = torch.load('C:/Users/zhk27/OneDrive/Рабочий стол/final_1/results/RAVDESS_multimodalcnn_15_best0.pth')
    model.load_state_dict(best_state['state_dict'])
    model.eval()
    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ans = np.array([])
    classification_prediction = np.empty((0,5))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, (inputs_audio, inputs_visual, targets) in enumerate(test_loader):
        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])

        targets = targets.to(opt.device)
        with torch.no_grad():
            inputs_visual = inputs_visual.to(device)
            inputs_audio = inputs_audio.to(device)
            targets = targets.to(device)
            outputs = model(inputs_audio, inputs_visual)
            classification_prediction = np.concatenate((classification_prediction,outputs.cpu().numpy()))
            _, preds = torch.max(outputs, 1)

            ans = np.concatenate((ans,preds.cpu().numpy())) 
            
    num_complete_groups = classification_prediction.shape[0] // 6
    reshaped_array = classification_prediction[:num_complete_groups * 6].reshape(num_complete_groups, 6, 5)
    
    probabilities = softmax(reshaped_array)

    
    probabilities[:,:, [0,1,2,3,4]] = probabilities[:,:, [0,3,4,2,1]]
   
    
    return probabilities, reshaped_array


