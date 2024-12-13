
import cv2
import numpy as np
import tensorflow as tf
import glob
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import keras
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import *
from matplotlib import pyplot as plt
seed_constant=27
tf.random.set_seed(seed_constant)
np.random.seed(seed_constant)
dirlist=['']
sequence_length=20
image_width=64
image_heigth=64
sport_list=["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
def video_frame_extraction(video_path):
    frame_list=[]
    video_capture=cv2.VideoCapture(video_path)
    for i in range (sequence_length):
        success,frame=video_capture.read()
        if not success:
            break
        
        resized_frame=cv2.resize(frame,(image_heigth,image_width))
        resized_frame=resized_frame/255
        frame_list.append(resized_frame)
       
    video_capture.release()
    return(frame_list)


def creat_dataset(main_folder_path):
    data=[]
    label_data=[]
    for sportname in sport_list:

        for i,file_path in enumerate(glob.glob(os.path.join(main_folder_path,sportname,"*"))):
            video_frame_list=video_frame_extraction(file_path)
            video_label=file_path.split("\\")[-2]
            if len(video_frame_list)==sequence_length:

                data.append(video_frame_list)
                label_data.append(video_label)
              
    data=np.array(data)
    label_data=np.array(label_data)


    return (data,label_data)
data,label_data=creat_dataset(r"C:\Users\VAIO\Downloads\Compressed\UCF50\UCF50")
print(len(data))
print(data[486].shape)
le=LabelEncoder()
label_data=le.fit_transform(label_data)
print  (label_data)
cat_labels=to_categorical(label_data)
print(cat_labels)
X_train,X_test,Y_train,Y_test=train_test_split(data,cat_labels,test_size=0.25,shuffle=True,random_state=seed_constant)

def LRCN():
    model=Sequential()
    model.add(TimeDistributed(Conv2D(16,(3,3),padding="same",activation="relu"),
    input_shape=(sequence_length,image_heigth,image_width,3)))
    model.add(TimeDistributed(MaxPooling2D(4,4)))
    model.add (TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(32,(3,3),padding="same",activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(4,4)))
    model.add (TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64,(3,3),padding="same",activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(2,2)))
    model.add (TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv2D(64,(3,3),padding="same",activation="relu")))
    model.add(TimeDistributed(MaxPooling2D(2,2)))
    model.add (TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add (LSTM(32))
    model.add(Dense(len(sport_list),activation="Softmax"))
    model.summary()
    return model

convlstm_model=LRCN()
convlstm_model.compile (loss='categorical_crossentropy',optimizer='Adam',metrics=["accuracy"])
history=convlstm_model.fit(x=X_train,y=Y_train,epochs=50,batch_size=4,shuffle=True,validation_split=0.2)
eval

model_eval_loss, model_eval_acc=convlstm_model.evaluate(X_test,Y_test)

convlstm_model.save(r"C:\Users\VAIO\Desktop\DSC\RNN\Week3_Complementary\code-VC\code-VC\video_LRCN_class.keras")
def plot_metrics (model_history,metric_name_1,metric_name_2,title):
   metric_value_1= model_history.history[metric_name_1]
   metric_value_2=model_history.history[metric_name_2]
   plt.plot(metric_value_1,label=metric_name_1)
   plt.plot(metric_value_2,label=metric_name_2)
   plt.title(title)
   plt.legend()
   plt.show()

plot_metrics(history,'accuracy','val_accuracy','Accuracy Plot')
plot_metrics(history,'loss','val_loss','Loss Plot')
print(model_eval_acc,"evaluation_accuracy")
print(model_eval_loss,"evaluation_loss")