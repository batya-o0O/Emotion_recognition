First install the requirements, actually there are more needed libraries that are indicated. Install all that needed. 

Then one should prepare the data as follows: root folder, inside of which each subject's folder. 

After that we need to first preprocess the data. Go to the ravdess_preprocessing folder.
First need to create annotations for data, to set which subjects are for training, validation and testing. These are needed to bet in create_annotation.py file. 
Then set the correct root directory in extract_audios and extract_faces files and run them. 

When the data is ready, you need to adjust opts.py file. Here we all different parameters, such as paths to annotations, path to a pretrained efficient_face model and etc. 
Change the paths for annotation.txt, results folder and for a model. Adjust other parameters for your needs. 

Then, you can run the main.py file which starts the training of the model. The results with best model will be saved inside the results folder. 

Now, you have an Audio-Video model. 

For EEG and Fusion we need to do this:  Inside of root folder create folders: validation and test. Then put the subjects in these folders. Subjects for training remains in the root folder.

For training the EEG model:

    Go to the eeg_all.py file, adjust all paths and run. 

For fusion:

   Go to the fusion_boosting.py file. Adjust the paths and run the file. 


ADjust the paths in prediction.py file 



We have 3 fusions first is weihted average, second is xgboost and third is custom CNN model. 
    Xgboost and CNN models were trained on validation data and tested on test data. The weigts in weighted average model were obtained from validation data. 

Inside of fusion_boosting we have all 3 fusions where each 21 seconds video is splitted into 6 parts. So model makes 6 prediction per video.


While inside of fusion_long we have almost the same 3 fusion, but here model makes only 1 prediction per 21 seconds video.