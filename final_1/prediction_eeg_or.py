import os
import scipy.io
import numpy as np
from tensorflow.keras.models import Model, load_model
from scipy import signal
from scipy.signal import butter



# Path to the root directory
parent_directory_ = r'C:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/'


# Insert the path to a model
model = load_model('C:/Users/zhk27/OneDrive/Рабочий стол/final_1/eeg_all_model.h5')
logits_model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Modify index as per your model architecture
# last_dense_layer = model.get_layer('dense')

# Get the weights of the last dense layer
# weights = last_dense_layer.get_weights()


# Define the Bandpass function
def Bandpass(dat, freq=[5, 80], fs=500):    
    [D, Ch, Tri] = dat.shape
    dat2 = np.transpose(dat, [0, 2, 1])
    dat3 = np.reshape(dat2, [10000*200, 30], order='F')
    
    sos = butter(5, freq, 'band', fs=fs, output='sos')  
    fdat = []
    for i in range(np.size(dat3, 1)):
        tm = signal.sosfilt(sos, dat3[:,i])
        fdat.append(tm)    
    dat4 = np.array(fdat).transpose().reshape((D, Tri, Ch), order='F').transpose(0, 2, 1)
    return dat4

# Define the mysplit function
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
def prediction_eeg(parent_directory):
    all_data =[]

    
    subject_folders = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and d.startswith("subject")]



    sorted_subject_folders = sorted(subject_folders, key=lambda s: int(s.replace("subject", "")))
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
        
        
        selected_classes = [1, 3, 5, 7, 9] # only speaking classes
        
        selected_indices = np.isin(np.argmax(Label_split, axis=0), selected_classes)

        data_5class = dat[selected_indices]
        
        aa = Label_split[:, selected_indices]
        label_5class = aa[selected_classes,:]
                
            
        # Lists to collect train and test subsets
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []
    
    
        for i in range(5):  # Looping over each class
            class_indices = np.where(label_5class.T[:, i] == 1)[0]  # Find indices where current class label is 1
            midpoint = len(class_indices)  # Calculate the midpoint for 50% split
        
        
            # this random shuffle will decide "random order" or "sequential order in time"
            # np.random.shuffle(class_indices)  # Shuffle the indices randomly

        
            # Split data based on found indices
            x_train_list.append(data_5class[class_indices[:midpoint]])
            x_test_list.append(data_5class[class_indices[midpoint:]])
        
            y_train_list.append(label_5class.T[class_indices[:midpoint]])
            y_test_list.append(label_5class.T[class_indices[midpoint:]])
        
        # Convert lists to numpy arrays
        x_train = np.concatenate(x_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        
        
        all_data.append(x_train)

    all_data_np = np.concatenate(all_data, axis=0)
    logits = logits_model.predict(all_data_np)
    print(logits.shape)

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)



    reshaped_array = logits.reshape(100*len(subject_folders), 6, 5)

    probabilities = softmax(reshaped_array)
    
    return probabilities, reshaped_array