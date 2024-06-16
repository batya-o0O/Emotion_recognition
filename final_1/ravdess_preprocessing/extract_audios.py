import librosa
import os
import soundfile as sf
import numpy as np

# -*- coding: utf-8 -*-


root = 'C:/Users/zhk27/OneDrive/Рабочий стол/Minho'
target_time = 3.6  # Target duration of each audio segment in seconds
step_time = 10/3  # Time interval between each segment in seconds

# Iterate over each subject in the root directory
for subject in os.listdir(root):
    subject_path = os.path.join(root, subject)
    
    # Skip if the item is not a directory
    if not os.path.isdir(subject_path):
        continue
    
    # Iterate over each audio file in the subject directory
    for audiofile in os.listdir(subject_path):
        
        # Skip if the file is not a WAV file or already processed
        if not audiofile.endswith('.wav') or 'croppad' in audiofile:
            continue
        
        # Load the audio file using librosa
        audios = librosa.core.load(os.path.join(subject_path, audiofile), sr=22050)
        y = audios[0]  # Audio samples
        sr = audios[1]  # Sampling rate
        
        # Calculate the target length and step length in samples
        target_length = int(sr * target_time)
        step_length = int(sr * step_time)
        
        # Extract segments from the audio file
        parts = audiofile.split('_')
        new_filename_base = '_'.join([parts[0], 'Trial'] + parts[3:-1])
        
        # Iterate over each segment in the audio file
        for i in range(0, len(y) - target_length + 1, step_length):
            segment = y[i:i + target_length]  # Extract the segment
            segment_filename = f"{new_filename_base}_{i // step_length}_croppad.wav"
            sf.write(os.path.join(subject_path, segment_filename), segment, sr)  # Save the segment as a new WAV file