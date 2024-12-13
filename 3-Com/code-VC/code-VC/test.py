import numpy as np
np.random.seed(12)
x=np.random.rand(3,4,4)

x=np.expand_dims(x)
print(x.shape)
print(x)

