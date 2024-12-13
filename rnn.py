import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.layers import SimpleRNN
from keras import Sequential
import tensorflow as tf
x=tf.random.normal((1,3,2))
print(x)
layer=SimpleRNN(4,input_shape=(3,2))
output=layer(x)
print(output.shape)
print(output)
#-------------------------------------------

import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
from keras.layers import SimpleRNN
from keras import Sequential
import tensorflow as tf
x=tf.random.normal((1,3,2))
print(x)
layer=SimpleRNN(4,input_shape=(None,2),return_sequences=True)
output=layer(x)
print(output.shape)

"""r3r3r2"""


print(output)
from keras.layers import Dense,TimeDistributed
model=Sequential()
model.add(SimpleRNN(4,input_shape=(3,2),return_sequences=True))
model.add(SimpleRNN(4,input_shape=(3,4),return_sequences=True))
model.add(TimeDistributed(Dense(6,activation="softmax")))
