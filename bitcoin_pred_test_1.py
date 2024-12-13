
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import pandas as pd
from keras import layers
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\VAIO\Desktop\DSC\RNN\Week3\BTC-USD2.csv")
print(data.head())
print(data.shape)
print(list(data.columns))

raw_data = data.iloc[:, 1:]
btc_price = data["mean"]

# plt.plot(range(len(btc_price)), btc_price)
# plt.show()

# plt.plot(range(1440), temperature[:1440])
# plt.show()
# """

num_train_samples = int(0.8 * len(raw_data))
# num_val_samples = int(0.25 * len(raw_data))
num_val_samples = len(raw_data) - num_train_samples

print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
# print("num_test_samples:", num_test_samples)

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
print(raw_data.shape)
exit

sampling_rate = 1
sequence_length = 60
predict_period_day=7
# delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 10
# print(raw_data.shape)
# input()
train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-predict_period_day],
    targets=btc_price[(predict_period_day+sequence_length-1):],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-predict_period_day],
    targets=btc_price[(predict_period_day+sequence_length-1):],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,      
    start_index=num_train_samples)

# test_dataset = keras.utils.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size,
#     start_index=num_train_samples + num_val_samples)

for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break   


inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.LSTM(16)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)



# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.LSTM(16, recurrent_dropout=0.25)(inputs)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)


# model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
# history = model.fit(train_dataset,
#                     epochs=10,
#                     validation_data=val_dataset)


# """



# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.GRU(32, recurrent_dropout=0.5, return_sequences=True)(inputs)
# x = layers.GRU(32, recurrent_dropout=0.5)(x)
# x = layers.Dropout(0.5)(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)


# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Bidirectional(layers.LSTM(16))(inputs)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)
# """

model.compile(optimizer="rmsprop", loss="mae", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset)


loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

