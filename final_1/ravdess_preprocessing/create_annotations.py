import os 

# Set the root directory
root = 'C:/Users/zhk27/OneDrive/Рабочий стол/Minho'

# Define a dictionary to map emotion labels to numbers
switcher={ 
    "Neutral": 1, 
    "Calmness": 2,  
    "Happiness": 3, 
    "Sadness": 4,  
    "Anger": 5 
} 

# Set the number of folds
n_folds=1 

# Define the fold IDs, first fold is used for testing, second fold is used for validation, and the remaining folds are used for training
folds = [[[41,40,39,38,37,36],[35,34,33,32,31,30],[29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]] 

# Iterate over each fold
for fold in range(n_folds): 
    fold_ids = folds[fold] 
    test_ids, val_ids, train_ids = fold_ids 

    # Set the annotation file name
    annotation_file = 'annotations.txt' 

    # Iterate over each actor in the root directory
    for i, actor in enumerate(os.listdir(root)): 
        # Iterate over each video in the actor's directory
        for video in os.listdir(os.path.join(root, actor)): 
            # Skip videos that are not in the .npy format or do not contain 'croppad' in their name
            if not video.endswith('.npy') or 'croppad' not in video: 
                continue 

            # Extract the label from the video name
            label_str = str(video.split('_')[3])
            label = str(switcher.get(label_str))

            # Create the paths for the video and audio files
            audio = video.split('_face')[0] + '_croppad.wav'    
            file_path = os.path.join(root, actor, video).replace('\\', '/') 
            audio_path = os.path.join(root, actor, audio).replace('\\', '/') 

            # Create the annotation line
            line = f"{file_path};{audio_path};{label};"

            # Determine if the video belongs to the training, validation, or testing set
            if i in train_ids: 
                line += "training" 
            elif i in val_ids: 
                line += "validation" 
            else: 
                line += "testing" 

            # Append the line to the annotation file
            with open(annotation_file, 'a') as f:                     
                f.write(line + '\n')