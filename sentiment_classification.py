import os
os.environ["TF_CPP_MIN_LOG_MIN_LEVEL"]="3"

from keras.utils import pad_sequences
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense,Embedding
from keras.layers import SimpleRNN

max_features=20000
maxlen=80
batch_size=32
print('loading data...')

(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print(len(x_train),"train sequences")
print(len(x_test),"test sequences")

print("Pad Sequences(Samples x time)")
x_train=pad_sequences(x_train,maxlen=maxlen)
x_test=pad_sequences(x_test,maxlen=maxlen)
print("x_train shape",x_train.shape)
print("x_test shape",x_test.shape)

print("build Model...")

Model=Sequential()
Model.add(Embedding(max_features,128))
Model.add(SimpleRNN(128))
Model.add(Dense(1,activation="sigmoid"))

Model.compile(loss='binary_crossentropy',optimizer="adam",metrics="accuracy")

print("train")
Model.fit(x_train,y_train,batch_size=batch_size,epochs=15,validation_data=(x_test,y_test))
loss,acc=Model.evaluate(x_test,y_test,batch_size=batch_size)
print("Test loss",loss)
print("Test accuracy",acc)