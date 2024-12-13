
from sklearn.model_selection import train_test_split
import random
import string
import tensorflow as tf
import re
from keras import *
random.seed(1000)
vocab_size=15000
sequence_length=20
batch_size=64
val_size=0.15
embed_dim=256
latent_dim=1024
def load_doc(path):
    with open (path, encoding='UTF-8') as f:
        lines=f.read().split("\n")[:-1]

    return(lines)
def create_pairs(lines):
    text_pairs=[]
    for line in lines:
        english, spanish=line.split("\t")
        spanish="[start] "+spanish+" [end]"
        text_pairs.append((english,spanish))
    print("[INFO]...Printing data sample...")
    print(random.choice(text_pairs))
    return(text_pairs)

def split_data(text_pairs):
    random.shuffle(text_pairs)
    num_val_data=int(len(text_pairs)*val_size)
    num_train_data=len(text_pairs)-(2*num_val_data)
    train_pairs=text_pairs[:num_train_data]
    val_pairs=text_pairs[num_train_data:num_train_data+num_val_data]
    test_pairs=text_pairs[num_train_data+num_val_data:]
    return(train_pairs,val_pairs,test_pairs)

def costum_standardization(input_string):
    strip_chars=string.punctuation+"¿"
    strip_chars=strip_chars.replace("[","")
    strip_chars=strip_chars.replace("]","")
    print(strip_chars,"**************************************************")
    lower_case=tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lower_case,f"[{re.escape(strip_chars)}]","")


def tokenization(train_pairs):

    source_tokenization=layers.TextVectorization(max_tokens=vocab_size,output_mode='int',output_sequence_length=sequence_length)
    target_tokenization=layers.TextVectorization(max_tokens=vocab_size,output_mode='int',output_sequence_length=sequence_length+1,standardize=costum_standardization)

    english_pairs=[train_pair[0] for train_pair in train_pairs ]
    spanish_pairs=[train_pair[1] for train_pair in train_pairs ]
    source_tokenization.adapt(english_pairs)
    target_tokenization.adapt(spanish_pairs)
    print ("[INFO] data tokenized and converted to int numbers" )
    return source_tokenization,target_tokenization
lines=load_doc(r"C:\Users\VAIO\Desktop\DSC\RNN\Week4\nmt example\nmt example\spa.txt")
text_pairs=create_pairs(lines)
train_pairs,val_pairs,test_pairs=split_data(text_pairs)
source_tokenization, target_tokenization=tokenization(train_pairs)

def format_dataset(eng,spa):
    eng=source_tokenization(eng)
    spa=target_tokenization(spa)
    return ({"english":eng,"spanish":spa[:,:-1]},spa[:,1:])
def make_dataset(pairs):
    eng_texts,spa_texts=zip(*pairs)
    eng_texts=list(eng_texts)
    spa_texts=list(spa_texts)
    dataset=tf.data.Dataset.from_tensor_slices((eng_texts,spa_texts))
    dataset=dataset.batch(batch_size)
    dataset=dataset.map(format_dataset,num_parallel_calls=4)
    return dataset.shuffle(2048)



source= layers.Input(shape=(None,),dtype="int64",name='english')
x=layers.Embedding(vocab_size,embed_dim,mask_zero=True)(source)
encoded_source= layers.Bidirectional(layers.GRU(latent_dim),merge_mode="sum")(x)

past_target=layers.Input(shape=(None,),dtype="int64",name='spanish')
y=layers.Embedding(vocab_size,embed_dim,mask_zero=True)(past_target)
y=layers.GRU(latent_dim,return_sequences=True)(y,initial_state=encoded_source)
y=layers.TimeDistributed(layers.Dropout(0.5))(y)
next_target_step=layers.TimeDistributed(layers.Dense(vocab_size,activation="softmax"))(y)
seq2seq_rnn=Model([source,past_target],next_target_step)

train_ds=make_dataset(train_pairs)
val_ds=make_dataset(val_pairs)

seq2seq_rnn.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])
seq2seq_rnn.fit(train_ds,epochs=15,validation_data=val_ds)




