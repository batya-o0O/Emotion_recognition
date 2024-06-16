# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:21:19 2023

@author: 504
"""
import os

import scipy.io
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz
from sklearn.metrics import accuracy_score
from scipy.linalg import eigh
import tensorflow as tf
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

'''
% TRIGGER_START_TRIAL_NEU = 44


% TRIGGER_START_TRIAL_NEU_LIS = 88, 0
% TRIGGER_START_TRIAL_NEU_SPE = 108, 1
% TRIGGER_START_TRIAL_S_LIS = 4, 2
% TRIGGER_START_TRIAL_S_SPE = 6, 3
% TRIGGER_START_TRIAL_A_LIS = 8, 4
% TRIGGER_START_TRIAL_A_SPE = 10, 5
% TRIGGER_START_TRIAL_H_LIS = 12, 6
% TRIGGER_START_TRIAL_H_SPE = 14, 7
% TRIGGER_START_TRIAL_R_LIS = 32, 8
% TRIGGER_START_TRIAL_R_SPE = 34, 9
% TRIGGER_START_TRIAL_NEU2 = 66, 10


A
A
R
R
S
S
H : (5*4) *: 80 tr
A
R
H : 20 4: 80 tr
H : 20 4: 80 40 tr 40 te
S
S
A
A
R
R
H : 20 4: 80 te
H : 20 4: 80 te
S

'''


# Path to the root directories


parent_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset'

# path for test and validation data folders
test_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/test'

val_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/validation'



# Get all directories in the parent directory that start with "subject"
subject_folders = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and d.startswith("subject")]
test_folders = [d for d in os.listdir(test_directory) if os.path.isdir(os.path.join(test_directory , d)) and d.startswith("subject")]
val_folders = [d for d in os.listdir(val_directory) if os.path.isdir(os.path.join(val_directory , d)) and d.startswith("subject")]



# Sort the list
sorted_subject_folders = sorted(subject_folders, key=lambda s: int(s.replace("subject", "")))
sorted_test_folders = sorted(test_folders, key=lambda s: int(s.replace("subject", "")))
sorted_val_folders = sorted(val_folders, key=lambda s: int(s.replace("subject", "")))




# Check if the sorted list is same as the original list
is_sorted = subject_folders == sorted_subject_folders



def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)



result_conf = list()
result_acc = list()
result_f1 = list()


all_data = []
all_labels = []

test_data = []
test_labels = []

val_data = []
val_labels = []


def Bandpass(dat, freq = [5, 80], fs = 500):    
        [D, Ch, Tri] = dat.shape
        dat2 = np.transpose(dat, [0, 2, 1])
        dat3 = np.reshape(dat2, [10000*200, 30], order = 'F')
        
        sos = butter(5, freq, 'band', fs=fs, output='sos')  
        fdat = list()
        for i in range(np.size(dat3, 1)):
            tm = signal.sosfilt(sos, dat3[:,i])
            fdat.append(tm)    
        dat4 =  np.array(fdat).transpose().reshape((D, Tri, Ch), order = 'F').transpose(0, 2, 1)
        return dat4
    
    # no need this function anymore
def seg(dat, marker, ival = [0, 2000]):    
    sdat = list()
    for i in range(np.size(marker, 1)):
        lag = range(marker[0, i]+ival[0], marker[0, i]+ival[1])
        sdat.append(dat[lag, :] )     
    return np.array(sdat)
    
    # data = (200, 2000, 30), label = (10, 160)
def mysplit(data, label): 
    total_length = data.shape[1]  # Total length in the second dimension
    split_length = 333  # Length of each split, exactly 333
    num_splits = total_length // split_length  # Number of full 333-length splits
    new_total_length = num_splits * split_length  # New total length that is a multiple of 333

    a1 = np.transpose(data, [1, 0, 2])
    
    # Reshape array by discarding extra data
    # Only include the first `new_total_length` data points
    a1_trimmed = a1[:new_total_length, :, :]

    # Reshape into the desired number of splits
    a2 = np.reshape(a1_trimmed, [split_length, num_splits, 200, 30], order='F')
    a3 = np.reshape(a2, [split_length, num_splits*200, 30], order='F')
    a4 = np.transpose(a3, [1, 0, 2])

    # Adjust labels similarly: repeat each label for the new number of splits
    labels_repeated = np.repeat(label, repeats=num_splits, axis=1)

    return a4, labels_repeated



for subject in sorted_subject_folders:
    # Remove trailing '__' if present
    # subject = subject.rstrip('__')
    cnt_ = []
    # Construct the full path to the EEG file
    eeg_folder = os.path.join(parent_directory, subject, 'EEG')
    eeg_file_name = subject.rstrip('__') + '_eeg.mat'
    eeg_file_path = os.path.join(eeg_folder, eeg_file_name)
    
    label_file_name = subject.rstrip('__') + '_eeg_label.mat'
    label_file_path = os.path.join(eeg_folder, label_file_name)

    # Check if the EEG file exists
    if os.path.exists(eeg_file_path):
        
        mat = scipy.io.loadmat(eeg_file_path)
        cnt_ = np.array(mat.get('seg1'))
        if np.ndim(cnt_) == 3:
            cnt_ = np.array(mat.get('seg1'))            
        else:
            cnt_ = np.array(mat.get('seg'))        
        
        mat_Y = scipy.io.loadmat(label_file_path)
        Label = np.array(mat_Y.get('label'))
        
        print(f'Loaded EEG data for {subject}')
    else:
        print(f'EEG data not found for {subject}')


    
    # input = (10000, 30, 200)
    # out = (10000, 30, 200)
    
    
    
    cnt_f = Bandpass(cnt_, freq = [3, 50], fs = 500)
    
    fs_original = 500  # Original sampling rate in Hz
    fs_target = 100  # Target sampling rate in Hz
    
    
    tm = np.transpose(cnt_f, [0, 2, 1]).reshape([10000*200, 30], order = 'F')
    
    downsampling_factor = fs_target / fs_original
    tm2 = signal.resample_poly(tm, up=1, down=int(fs_original/fs_target), axis=0)
    cnt_f2= np.reshape(tm2, [2000, 200, 30], order = 'F')
    
    cnt_seg = np.transpose(cnt_f2, [1, 0, 2])
    
    num_trials_per_class = np.sum(Label, axis=1) # check the balance. 
    [cnt_seg_split, Label_split]= mysplit(cnt_seg, Label)
    
    
    dat = np.transpose(cnt_seg_split, (0, 2, 1)).reshape((1200, 30, 333, 1)) # This should be (800, 30, 500, 1) for balanced classes
    
    
    selected_classes = [0, 2, 4, 6, 8] # only speaking classes
    
    selected_indices = np.isin(np.argmax(Label_split, axis=0), selected_classes)

    data_5class = dat[selected_indices]
    
    aa = Label_split[:, selected_indices]
    label_5class = aa[selected_classes,:]
    
    #x_train, x_test, y_train, y_test = train_test_split(data_5class, label_5class.T, test_size=0.5, random_state=42, stratify=label_5class.T)
    
        
    # Lists to collect train and test subsets
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    
    
    for i in range(5):  # Looping over each class
        class_indices = np.where(label_5class.T[:, i] == 1)[0]  # Find indices where current class label is 1
        midpoint = len(class_indices)  # Calculate the midpoint for 50% split
    
        # Split data based on found indices
        x_train_list.append(data_5class[class_indices[:midpoint]])
        x_test_list.append(data_5class[class_indices[midpoint:]])
    
        y_train_list.append(label_5class.T[class_indices[:midpoint]])
        y_test_list.append(label_5class.T[class_indices[midpoint:]])
    
    # Convert lists to numpy arrays
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    all_data.append(x_train)
    all_labels.append(y_train)
    
    

for subject in sorted_test_folders:
    # Remove trailing '__' if present
    # subject = subject.rstrip('__')
    cnt_ = []
    # Construct the full path to the EEG file
    eeg_folder = os.path.join(test_directory, subject, 'EEG')
    eeg_file_name = subject.rstrip('__') + '_eeg.mat'
    eeg_file_path = os.path.join(eeg_folder, eeg_file_name)
    
    label_file_name = subject.rstrip('__') + '_eeg_label.mat'
    label_file_path = os.path.join(eeg_folder, label_file_name)

    # Check if the EEG file exists
    if os.path.exists(eeg_file_path):
        
        mat = scipy.io.loadmat(eeg_file_path)
        cnt_ = np.array(mat.get('seg1'))
        if np.ndim(cnt_) == 3:
            cnt_ = np.array(mat.get('seg1'))            
        else:
            cnt_ = np.array(mat.get('seg'))        
        
        mat_Y = scipy.io.loadmat(label_file_path)
        Label = np.array(mat_Y.get('label'))
        
        print(f'Loaded EEG data for {subject}')
    else:
        print(f'EEG data not found for {subject}')


    
    # input = (10000, 30, 200)
    # out = (10000, 30, 200)
    
    
    
    cnt_f = Bandpass(cnt_, freq = [3, 50], fs = 500)
    
    fs_original = 500  # Original sampling rate in Hz
    fs_target = 100  # Target sampling rate in Hz
    
    
    tm = np.transpose(cnt_f, [0, 2, 1]).reshape([10000*200, 30], order = 'F')
    
    downsampling_factor = fs_target / fs_original
    tm2 = signal.resample_poly(tm, up=1, down=int(fs_original/fs_target), axis=0)
    cnt_f2= np.reshape(tm2, [2000, 200, 30], order = 'F')
    
    cnt_seg = np.transpose(cnt_f2, [1, 0, 2])
    
    num_trials_per_class = np.sum(Label, axis=1) # check the balance. 
    [cnt_seg_split, Label_split]= mysplit(cnt_seg, Label)
    
    
    dat = np.transpose(cnt_seg_split, (0, 2, 1)).reshape((1200, 30, 333, 1))  # This should be (800, 30, 500, 1) for balanced classes
    
    
    selected_classes = [0, 2, 4, 6, 8] # only listening classes
    
    selected_indices = np.isin(np.argmax(Label_split, axis=0), selected_classes)

    data_5class = dat[selected_indices]
    
    aa = Label_split[:, selected_indices]
    label_5class = aa[selected_classes,:]
    
    #x_train, x_test, y_train, y_test = train_test_split(data_5class, label_5class.T, test_size=0.5, random_state=42, stratify=label_5class.T)
    
        
    # Lists to collect train and test subsets
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    
    
    for i in range(5):  # Looping over each class
        class_indices = np.where(label_5class.T[:, i] == 1)[0]  # Find indices where current class label is 1
        midpoint = len(class_indices)  # Calculate the midpoint for 50% split

        # Split data based on found indices
        x_train_list.append(data_5class[class_indices[:midpoint]])
        x_test_list.append(data_5class[class_indices[midpoint:]])
    
        y_train_list.append(label_5class.T[class_indices[:midpoint]])
        y_test_list.append(label_5class.T[class_indices[midpoint:]])
    
    # Convert lists to numpy arrays
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    test_data.append(x_train)
    test_labels.append(y_train)





for subject in sorted_val_folders:
    # Remove trailing '__' if present
    # subject = subject.rstrip('__')
    cnt_ = []
    # Construct the full path to the EEG file
    eeg_folder = os.path.join(val_directory, subject, 'EEG')
    eeg_file_name = subject.rstrip('__') + '_eeg.mat'
    eeg_file_path = os.path.join(eeg_folder, eeg_file_name)
    
    label_file_name = subject.rstrip('__') + '_eeg_label.mat'
    label_file_path = os.path.join(eeg_folder, label_file_name)

    # Check if the EEG file exists
    if os.path.exists(eeg_file_path):
  
        mat = scipy.io.loadmat(eeg_file_path)
        cnt_ = np.array(mat.get('seg1'))
        if np.ndim(cnt_) == 3:
            cnt_ = np.array(mat.get('seg1'))            
        else:
            cnt_ = np.array(mat.get('seg'))        
        
        mat_Y = scipy.io.loadmat(label_file_path)
        Label = np.array(mat_Y.get('label'))
        
        print(f'Loaded EEG data for {subject}')
    else:
        print(f'EEG data not found for {subject}')


    
    # input = (10000, 30, 200)
    # out = (10000, 30, 200)
    
    
    
    cnt_f = Bandpass(cnt_, freq = [3, 50], fs = 500)
    
    fs_original = 500  # Original sampling rate in Hz
    fs_target = 100  # Target sampling rate in Hz
    
    
    tm = np.transpose(cnt_f, [0, 2, 1]).reshape([10000*200, 30], order = 'F')
    
    downsampling_factor = fs_target / fs_original
    tm2 = signal.resample_poly(tm, up=1, down=int(fs_original/fs_target), axis=0)
    cnt_f2= np.reshape(tm2, [2000, 200, 30], order = 'F')
    
    cnt_seg = np.transpose(cnt_f2, [1, 0, 2])
    
    num_trials_per_class = np.sum(Label, axis=1) # check the balance. 
    [cnt_seg_split, Label_split]= mysplit(cnt_seg, Label)
    
    
    dat = np.transpose(cnt_seg_split, (0, 2, 1)).reshape((1200, 30, 333, 1))  # This should be (800, 30, 500, 1) for balanced classes
    
    
    selected_classes = [0, 2, 4, 6, 8] # only listening classes
    
    selected_indices = np.isin(np.argmax(Label_split, axis=0), selected_classes)

    data_5class = dat[selected_indices]
    
    aa = Label_split[:, selected_indices]
    label_5class = aa[selected_classes,:]
    
    #x_train, x_test, y_train, y_test = train_test_split(data_5class, label_5class.T, test_size=0.5, random_state=42, stratify=label_5class.T)
    
        
    # Lists to collect train and test subsets
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    
    
    for i in range(5):  # Looping over each class
        class_indices = np.where(label_5class.T[:, i] == 1)[0]  # Find indices where current class label is 1
        midpoint = len(class_indices)  # Calculate the midpoint for 50% split
    

        # Split data based on found indices
        x_train_list.append(data_5class[class_indices[:midpoint]])
        x_test_list.append(data_5class[class_indices[midpoint:]])
    
        y_train_list.append(label_5class.T[class_indices[:midpoint]])
        y_test_list.append(label_5class.T[class_indices[midpoint:]])
    
    # Convert lists to numpy arrays
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    val_data.append(x_train)
    val_labels.append(y_train)



    


all_data_np = np.concatenate(all_data, axis=0)
all_labels_np = np.concatenate(all_labels, axis=0)

test_data_np = np.concatenate(test_data, axis=0)
test_labels_np = np.concatenate(test_labels, axis=0)

val_data_np = np.concatenate(val_data, axis=0)
val_labels_np = np.concatenate(val_labels, axis=0)


model = EEGNet(nb_classes = 5, D = 8, F2 = 64, Chans = 30, kernLength = 300, Samples = 333, 
                   dropoutRate = 0.5)
    
    
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

history = model.fit(all_data_np , all_labels_np, batch_size = 32, epochs = 50,shuffle=True,  validation_data = (val_data_np , val_labels_np ))        

pred = model.predict(test_data_np )    
pred = np.argmax(pred, axis=1)    

y_test2 = np.argmax(test_labels_np , axis=1)

cm = confusion_matrix(pred, y_test2)

accuracy = accuracy_score(pred, y_test2)
f1 = f1_score(y_test2, pred, average='weighted')  # 'weighted' for multiclass F1-Score
 
result_conf_np = np.array(result_conf)
summed_confusion_matrix = np.sum(result_conf_np, axis=0)
summed_confusion_matrix_T = summed_confusion_matrix.T


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print(summed_confusion_matrix_T)

print(accuracy)

# Insert your path for saving the model
model_path = 'C:/Users/zhk27/OneDrive/Рабочий стол/final_1/eeg_all_model.h5'
model.save(model_path)






