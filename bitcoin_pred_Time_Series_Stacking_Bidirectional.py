# Bitcoin Prediction#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import timeseries_dataset_from_array
from keras.layers import LSTM,Dense,Dropout,Bidirectional
from keras import Model
import keras
df=pd.read_csv(r"C:\Users\VAIO\Desktop\DSC\RNN\Week3\BTC-USD2.csv")
print(df.head())
print(df.shape)
print(list(df.columns))


btcprice=df['mean']
raw_data=df.iloc[:,1:]
plt.plot(range(len(btcprice)),btcprice)
# plt.show()
num_row_d=len(btcprice)
num_train_d=int(num_row_d*0.8)
num_test_d=num_row_d-num_train_d
print(num_train_d,'train_data')
print(num_test_d,'test_data')
mean=raw_data[:num_train_d].mean(axis=0)
raw_data-=mean
std=raw_data[:num_train_d].std(axis=0)
raw_data/=std

train_raw_data=raw_data.iloc[:num_train_d,:]
test_raw_data=raw_data.iloc[num_train_d:,]
train_btc_price=btcprice.iloc[:num_train_d,]
test_btc_price=btcprice.iloc[num_train_d:,]
sequence_length=60
print(test_btc_price.shape,'sHAPE')
predict_period=7
batch_size=20
train_time_series_dataset=timeseries_dataset_from_array(
    data=train_raw_data[:-7]
                                                  ,targets=train_btc_price[(predict_period+sequence_length-1):]
                                                  ,sequence_length=sequence_length
                                                  ,shuffle=True,batch_size=batch_size)
validation_time_series_dataset=timeseries_dataset_from_array(data=test_raw_data[:-predict_period]
                                                  ,targets=test_btc_price[(sequence_length+predict_period):]
                                                  ,sequence_length=sequence_length,shuffle=True,batch_size=batch_size)


# for i,(input ,target) in enumerate(train_time_series_dataset):
#     print(input.shape,"input")
#     print(target.shape,"target")
#     if i==5:
#         break
# input=keras.Input(shape=(sequence_length,train_raw_data.shape[-1]))
# x=(LSTM(32,recurrent_dropout=0.25)(input))
# x=Dropout(0.5)(x)
# output=Dense(1)(x)
# model=Model(input,output)
# model.compile(optimizer="adam",loss='mse',metrics=['mae'])
# history=model.fit(train_time_series_dataset,epochs=100,validation_data=validation_time_series_dataset)


# inam harmah ba Bidirectional RNN emtehan konam bebinam vase Time Series Che javabi mide"
input=keras.Input(shape=(sequence_length,train_raw_data.shape[-1]))
x=Bidirectional(LSTM(32,return_sequences=True))(input)
x=Bidirectional(LSTM(64))(x)
output=Dense(1)(x)
model=Model(input,output)
model.compile(optimizer="adam",loss='mse',metrics=['mae'])
history=model.fit(train_time_series_dataset,epochs=100,validation_data=validation_time_series_dataset)




loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

# print(train_raw_data.head())
# print(test_btc_price.head())



# raw_data=df.iloc[:,]
# -----------------------
# video hafte 3- bakhshe 2- dghighe 7:19