from keras.utils import pad_sequences
sequence=[[0],[1,2,3,4],[1,2],[3]]
ss=pad_sequences(sequence,maxlen=2)

from keras.layers import Embedding
print(ss)

ff=Embedding(5,10)

kk=ff(ss)
print(kk)