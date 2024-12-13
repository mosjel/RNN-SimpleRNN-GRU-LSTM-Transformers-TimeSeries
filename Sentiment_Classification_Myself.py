from keras.datasets import imdb
import numpy as np
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding
from keras.utils import pad_sequences
max_features=20000
max_len=80
batch_size=32
(x_train,y_train),(x_test,y_test),=imdb.load_data(num_words=max_features)
print(y_train)
print(type(y_train))
print(x_train.dtype)
print(x_train.shape)
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)
model=Sequential()
model.add(Embedding(max_features,128))
model.add(SimpleRNN(128))
model.add(Dense(1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics="accuracy")
H=model.fit(x_train,y_train,batch_size=batch_size,epochs=15,validation_data=(x_test,y_test))


loss,acc=model.evaluate(x_test,y_test,batch_size=batch_size)

print("test data loss",loss)
print("test data accuracy",acc)
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.plot(H.history["accuracy"],"train_data Accuracy")
plt.plot(H.history["val_accuracy"],"test_data Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("IMDB DATASET")
plt.show()

