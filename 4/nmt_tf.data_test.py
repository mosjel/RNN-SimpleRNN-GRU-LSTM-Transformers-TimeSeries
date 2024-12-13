from tensorflow import data
import numpy as np
random_numbers=np.random.normal(size=(1000,16))
s=data.Dataset.from_tensor_slices(random_numbers)
batched_data=s.batch(16)
for i,data1 in enumerate(batched_data):
    print(data1)
    if i==2: 
        break