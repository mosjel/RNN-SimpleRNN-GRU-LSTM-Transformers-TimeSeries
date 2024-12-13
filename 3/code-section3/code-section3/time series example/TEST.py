from keras.utils import timeseries_dataset_from_array
s=list(range(0,100))
print(s)
sampling_rate = 6
sequence_length = 5
delay = sampling_rate * (sequence_length + 1 - 1)
batch_size = 10
train_dataset = timeseries_dataset_from_array(
    s[:-(delay-1)],
    targets=s[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    
    batch_size=batch_size,
    start_index=0,
    end_index=60)

for samples, targets in train_dataset:
    print("samples :", samples.shape)
    print("targets :", targets.shape)
   