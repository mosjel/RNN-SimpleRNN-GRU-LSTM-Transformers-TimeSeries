import numpy as np
f=np.random.rand(10,4,20)
print(f)
import keras

s=keras.layers.LSTM(4)(f)
print(s)

