from prediction import prediction
from prediction_eeg_or import prediction_eeg
import os 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

test_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/test'
val_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/validation'

av_pred_test, av_cnn_test=prediction(lateFusion=True, test = True)
eeg_pred_test, eeg_cnn_test =prediction_eeg(test_directory)

av_pred_test = np.sum(av_pred_test, axis=1)
av_pred_test /= 6
         
eeg_pred_test = np.sum(eeg_pred_test, axis=1)
eeg_pred_test /= 6
   

av_pred_val, av_cnn_train = prediction(lateFusion=True, test = False)
eeg_pred_val, eeg_cnn_train = prediction_eeg(val_directory)

av_pred_val = np.sum(av_pred_val, axis=1)
av_pred_val /= 6
         
eeg_pred_val = np.sum(eeg_pred_val, axis=1)
eeg_pred_val /= 6

# eeg_pred_train, eeg_cnn_train_all = prediction_eeg(train_directory)
# eeg_pred_val = np.concatenate((eeg_pred_train, eeg_pred_val))
# eeg_cnn_train = np.concatenate((eeg_cnn_train_all, eeg_cnn_train))



res=av_pred_test*0.61+eeg_pred_test*0.39
print(res.shape)

preds = np.argmax(res, axis=1)


switcher={ 
        "Neutral": 0, 
        "Calmness": 4,  
        "Happiness": 3, 
        "Sadness": 1,  
        "Anger": 2
    } 

root = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/validation'


labels=[]
for subject in os.listdir(os.path.join(root)):
    for sess in os.listdir(os.path.join(root,subject)):
        if "Audio" in sess:
            for audio in os.listdir(os.path.join(root,subject,sess)):  
                label_str = str(audio.split('_')[4])
                label = switcher.get(label_str)
                labels.append(label)


class_names = ['Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness']

# Generate the classification report
report = classification_report(labels, preds, target_names=class_names)
#print(report)

def accuracy_score(labels, preds):
    correct_predictions = np.sum(labels == preds)
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions
    return accuracy 

accuracy = accuracy_score(labels, preds)
#print(f"Accuracy: {accuracy:.2f}")


class_names = sorted(switcher, key=switcher.get)
cm = confusion_matrix(labels, preds, labels=[switcher[class_name] for class_name in class_names])

    

stacked_predictions = np.hstack((av_pred_val, eeg_pred_val))
test_stacked = np.hstack((av_pred_test, eeg_pred_test))


root = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/test'


test_labels=[]
for subject in os.listdir(os.path.join(root)):
    for sess in os.listdir(os.path.join(root,subject)):
        if "Audio" in sess:
            for audio in os.listdir(os.path.join(root,subject,sess)):  
                label_str = str(audio.split('_')[4])
                label = switcher.get(label_str)
                test_labels.append(label)





res=av_pred_test*0.61+eeg_pred_test*0.39

preds = np.argmax(res, axis=1)
report = classification_report(labels, preds, target_names=class_names)
print(report)




train_stacked = stacked_predictions
train_labels = labels

print(train_stacked.shape)
print(len(labels))

# Train meta-model
model_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42, objective='multi:softprob', num_class=len(set(labels)))
model_xgb.fit(train_stacked, train_labels)

# Make final predictions on test data
final_predictions = model_xgb.predict(test_stacked)
final_accuracy = accuracy_score(test_labels, final_predictions)
print("Accuracy of meta-model: {:.2f}%".format(final_accuracy * 100))

# Print and plot classification report and confusion matrix
print(classification_report(test_labels, final_predictions, target_names=class_names))
cm = confusion_matrix(test_labels, final_predictions, labels=[switcher[class_name] for class_name in sorted(switcher, key=switcher.get)])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(switcher, key=switcher.get), yticklabels=sorted(switcher, key=switcher.get))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()








labels = to_categorical(labels, num_classes=5)
test_labels = to_categorical(test_labels, num_classes=5)

cnn_train = np.concatenate((av_cnn_train, eeg_cnn_train), axis=1)
cnn_test =  np.concatenate((av_cnn_test, eeg_cnn_test), axis=1)

print(av_cnn_train.shape)
print(eeg_cnn_train.shape)
print(cnn_train.shape)

cnn_train = cnn_train.reshape((-1, 12, 5, 1))  # Reshape to (600, 12, 5, 1)
cnn_test = cnn_test.reshape((-1, 12, 5, 1))    # 

print(cnn_train.shape)


#Model configuration
input_shape = (12, 5, 1)
num_classes = 5
dropout_rate = 0.5
l2_lambda = 0.01

# Building the model
model_cnn = Sequential([
    Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(dropout_rate),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(dropout_rate),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
    Dropout(dropout_rate),
    Dense(num_classes, activation='softmax')
])

# Model summary
model_cnn.summary()

# Compile the model
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation data
history=model_cnn.fit(cnn_train, labels, batch_size=64, epochs=50, validation_data=(cnn_test, test_labels))

# Evaluate the model on the testing data
test_loss, test_accuracy = model_cnn.evaluate(cnn_test, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

# Plotting training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'go-', label='Testing accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#model_cnn.save('c:/Users/zhk27/OneDrive/Рабочий стол/eeg/cnn_fusion.h5')


predictions = model_cnn.predict(cnn_test)

# Convert predictions from probabilities to class labels
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Define your class names


# Generate the classification report
report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)
# loaded_model = load_model('my_model.h5')

# # Assuming 'new_data' is preprocessed and ready for prediction
# predictions = loaded_model.predict(new_data)
# predicted_labels = np.argmax(predictions, axis=1)