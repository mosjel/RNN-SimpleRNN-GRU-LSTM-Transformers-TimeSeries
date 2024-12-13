import warnings
warnings.filterwarnings("ignore")
import pandas as pd

train_df=pd.read_csv(r"C:\Users\VAIO\Desktop\DSC\RNN\Week9\Notebooks\Notebooks\fine tuning\train.csv",usecols=["id","text","target"])

test_df=pd.read_csv(r"C:\Users\VAIO\Desktop\DSC\RNN\Week9\Notebooks\Notebooks\fine tuning\test.csv",usecols=["id","text"])

print(train_df.head())
print(test_df.head())
print("-----------------")
print(train_df.target)

# import text_hammer as th
# def text_processing(df,col_name):
#     column=col_name
#     df[column]=df[column].apply(lambda x:str(x).lower())
#     df[column]=df[column].apply(lambda x:th.remove_emails(x))
#     df[column]=df[column].apply(lambda x:th.remove_especial_chars(x))
#     df[column]=df[column].apply(lambda x:th.remove_accented_chars(x))
#     return(df)

# train_cleaned_df=text_processing(train_df,"text")
# train_cleaned_df[train_cleaned_df.target==0]
# train_df=train_cleaned_df.copy()
print ("Max number of word in every tweet is: ",max([len(i.split()) for i in train_df.text ]))
from transformers import BertTokenizer,TFBertModel
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
bert_model=TFBertModel.from_pretrained("bert-base-uncased")

x_train=tokenizer(
    text=train_df.text.tolist(),
    padding=True,
    truncation=True
    max_length=36
    return_tensors="tf"

)
x_train["input_ids"].shape
x_train["attention_mask"].shape

y_test=train_df.target.values()
print(y_test)

import tensorflow as tf
from keras import layers
from keras.optimizers import Adam
max_length=36
input_ids=layers.Input(shape=(max_length,),dtype=tf.int32,name="input_ids")
input_mask=layers.Input(shape=(max_length,),dtype=tf.int32,name="attention_mask")
embeddings=bert_model(input_ids,attention_mask=input_mask)[1] #(0 is last hidden states, 1 means pooler_output)

out=layers.Dropout(0.1)(embeddings)
out=layers.Dense(128,activation="relu")(out)
out=layers.Dense(0.1)(out)
out=layers.Dense(32,activation="relu")(out)
y=layers.Dense(1,activation="sigmoid")(out)

model=tf.keras.Model(inputs=[input_ids,input_mask],outputs=y)
model.layers[2].trainable=True
model.summary()

optimizer=Adam(learning_rate=6e-06,
               epsilon=1e-08,
               decay=0.01)
model.compile(optimizer= optimizer
              loss="binary_crossentropy"
              metrics=["accuracy"])
train_history=model.fit(x={"input_ids":x_train["input_ids"],"attention_mask":x_train["attention_mask"]}
                        y=y_train
                    validation_split=0.1
                    epochs=2,
                    batch_size=32)
x_test=tokenizer(text=test_df.text.tolist(),
                 padding=True,
                 max_length=36,
                 truncation=True,
                 return_tensors="tf"
                 )
predicted=model.predict(x={"input_ids":x_test["input_ids"],"attention_mask":x_test["attention_mask"]})



