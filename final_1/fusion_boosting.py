import os 
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

from prediction import prediction  
from prediction_eeg_or import prediction_eeg 

test_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/test'
val_directory = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/validation'


av_pred_test, av_cnn_test = prediction(lateFusion=True, test = True)
eeg_pred_test, eeg_cnn_test =prediction_eeg(test_directory)

av_pred_val, av_cnn_train = prediction(lateFusion=True, test = False)
eeg_pred_val, eeg_cnn_train = prediction_eeg(val_directory)



res=av_pred_test*0.61+eeg_pred_test*0.365
print(res.shape)
res=res.reshape((3600, 5))
preds = np.argmax(res, axis=1)


switcher={ 
        "Neutral": 0, 
        "Calmness": 4,  
        "Happiness": 3, 
        "Sadness": 1,  
        "Anger": 2
    } 

val_root = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/validation'

labels=[]
for subject in os.listdir(os.path.join(val_root )):
    for sess in os.listdir(os.path.join(val_root ,subject)):
        if "Audio" in sess:
            for audio in os.listdir(os.path.join(val_root ,subject,sess)):  
                label_str = str(audio.split('_')[4])
                label = switcher.get(label_str)
                labels.append(label)
                labels.append(label)
                labels.append(label)
                labels.append(label)
                labels.append(label)
                labels.append(label)
       
       
               
class_names = ['Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness']

report = classification_report(labels, preds, target_names=class_names)
print(report)

def accuracy_score(labels, preds):
    correct_predictions = np.sum(labels == preds)
    total_predictions = len(labels)
    accuracy = correct_predictions / total_predictions
    return accuracy 

accuracy = accuracy_score(labels, preds)
print(f"Accuracy: {accuracy:.2f}")

    

stacked_predictions = np.hstack((av_pred_val.reshape((3600,5)), eeg_pred_val.reshape((3600,5))))
test_stacked = np.hstack((av_pred_test.reshape((3600,5)), eeg_pred_test.reshape((3600,5))))




test_root = 'c:/Users/zhk27/OneDrive/Рабочий стол/Minho_Lee_Dataset/test'

test_labels=[]
for subject in os.listdir(os.path.join(test_root )):
    for sess in os.listdir(os.path.join(test_root ,subject)):
        if "Audio" in sess:
            for audio in os.listdir(os.path.join(test_root ,subject,sess)):  
                label_str = str(audio.split('_')[4])
                label = switcher.get(label_str)
                test_labels.append(label)
                test_labels.append(label)
                test_labels.append(label)
                test_labels.append(label)
                test_labels.append(label)
                test_labels.append(label)


train_stacked = stacked_predictions
train_labels = labels

print(train_stacked.shape)
print(len(labels))

model_xgb = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42, objective='multi:softprob', num_class=len(set(labels)))
model_xgb.fit(train_stacked, train_labels)

final_predictions = model_xgb.predict(test_stacked)
final_accuracy = accuracy_score(test_labels, final_predictions)
print("Accuracy of meta-model: {:.2f}%".format(final_accuracy * 100))

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

cnn_train = np.concatenate((av_cnn_train.reshape((3600,1,5)), eeg_cnn_train.reshape((3600,1,5))), axis=1)
cnn_test =  np.concatenate((av_cnn_test.reshape((3600,1,5)), eeg_cnn_test.reshape((3600,1,5))), axis=1)

cnn_train = cnn_train.reshape((-1, 2, 5, 1))  
cnn_test = cnn_test.reshape((-1, 2, 5, 1))    

print(cnn_train.shape)


input_shape = (2, 5, 1)
num_classes = 5
dropout_rate = 0.5
l2_lambda = 0.01

model_cnn = Sequential([
    Conv2D(32, kernel_size=(2, 2), padding="same", activation='relu', input_shape=input_shape),
    Dropout(dropout_rate),
    Conv2D(64, kernel_size=(2, 2), padding="same", activation='relu'),
    Dropout(dropout_rate),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda)),
    Dropout(dropout_rate),
    Dense(num_classes, activation='softmax')
])

model_cnn.summary()

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history=model_cnn.fit(cnn_train, labels, batch_size=64, epochs=50, validation_data=(cnn_test, test_labels))

test_loss, test_accuracy = model_cnn.evaluate(cnn_test, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'go-', label='Testing accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


cnn_model_output_paht = 'C:/Users/zhk27/OneDrive/Рабочий стол/final_1/cnn_fusion.h5'
model_cnn.save(cnn_model_output_paht)

predictions = model_cnn.predict(cnn_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

report = classification_report(true_classes, predicted_classes, target_names=class_names)
print(report)
