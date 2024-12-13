from keras.layers import Dropout
import numpy as np
import tensorflow as tf

np.random.seed(2)
x=np.random.uniform(0,1,(10,5))
y=np.random.choice(range(1,13),(4,3),replace=False).astype(float)
print(y)
# print (x)
tf.random.set_seed(0)
layer_=Dropout(0.4,input_shape=(3,))
s=layer_(y,training=True)
print(s)