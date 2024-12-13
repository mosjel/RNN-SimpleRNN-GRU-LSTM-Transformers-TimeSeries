import os 
os.environ['TF_ CPP_MIN_LOG_LEVEL']='3'

import pandas as pd
import glob 
from keras.utils import text_dataset_from_directory
from keras import layers
import keras
batch_size=32

train_ds=text_dataset_from_directory(r"C:\Users\VAIO\Desktop\DSC\RNN\Week2\section2-code\sentiment classification\aclImdb\train",batch_size=batch_size)
test_ds=text_dataset_from_directory(r"C:\Users\VAIO\Desktop\DSC\RNN\Week2\section2-code\sentiment classification\aclImdb\test",batch_size=batch_size)

for inputs,targets in train_ds:
    print(inputs.shape)
    print(inputs.dtype)
    print(targets.shape)
    print(targets.dtype)
    print("inputs[0]",inputs[31])
    print("targets[0]",targets[31])
    print(targets)
    break
text_only_train_ds=train_ds.map(lambda x,y:x)
max_tokens=20000
max_word=600
text_vectorization=layers.TextVectorization(
    output_sequence_length=max_word,max_tokens=max_tokens,output_mode="int"
    )
text_vectorization.adapt(text_only_train_ds)
int_train_ds=train_ds.map(lambda x,y:(text_vectorization(x),y),num_parallel_calls=4)
int_test_ds=test_ds.map(lambda x,y:(text_vectorization(x),y),num_parallel_calls=4)

inputs=keras.Input(shape=(None,),dtype="int64")
embedded=layers.Embedding(input_dim=max_tokens,output_dim=256)(inputs)
x=layers.LSTM(32)(embedded)
outputs=layers.Dense(1,activation="sigmoid")(x)
model=keras.Model(inputs,outputs)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])
model.summary()
model.fit(int_train_ds,validation_data=(int_test_ds),epochs=10)



