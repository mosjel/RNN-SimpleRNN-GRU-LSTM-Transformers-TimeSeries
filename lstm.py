import tensorflow as tf
from keras.layers import LSTM
tf.random.set_seed(12)
s=tf.random.normal([32,10,8])

r=LSTM(4)
result=r(s)
print(result)

"-------------------------------"
r1=LSTM(4,return_sequences=True,return_state=True)


whole_seq_output,final_memory_state,final_carry_state=r1(s)
print("whole_seq_output",whole_seq_output)
print("final_memory_state",final_memory_state)
print("final_carry_state",final_carry_state)
