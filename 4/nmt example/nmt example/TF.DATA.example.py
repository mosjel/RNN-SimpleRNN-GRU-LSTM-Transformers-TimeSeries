import tensorflow as tf
import numpy as np
batch_size=32
s=np.random.normal(size=(1000,16))
print(s.shape)
dataset=tf.data.Dataset.from_tensor_slices(s)
dataset=dataset.batch(batch_size)
print(dataset)
print(len(dataset))
dataset1=dataset.map(lambda x:tf.reshape(x,(_,4,4)))
for i,element in enumerate(dataset):
    print(element.shape)
    print(i)


