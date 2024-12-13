from transformers import BertTokenizer,TFBertModel
import os
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=TFBertModel.from_pretrained("bert-base-uncased")
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

inputs=tokenizer("Robotech Academy",return_tensors="tf")

result=model(inputs)

print(result.last_hidden_state.shape)
print(result.pooler_output.shape)
