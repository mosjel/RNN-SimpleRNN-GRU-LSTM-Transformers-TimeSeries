import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
# persian_text="من حامد هستم ".encode("utf-8")
# encoded_input=tokenizer("robotech academy!")
# print (encoded_input)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
batch_sentences=["But what abut second breakfast?","Don't think he knows about second breakfast, Pip.","What about elevensies?"]

# encoded_input=tokenizer(batch_sentences,padding=True,truncation=True,max_length=3,return_tensors="tf")
encoded_input=tokenizer(batch_sentences,padding=True,truncation=True,return_tensors="tf")
print(encoded_input)

encoded_input=tokenizer("Robotech Academy!")

decoded_input=tokenizer.decode(encoded_input["input_ids"])
print(decoded_input)



# batch_sentences=["Hello i am a single sentence","And another sentence","And the very very last one"]
# batch_of_second_sentences=["I'm a sentence that goes with the first sentence","And I should be encoded with the second sentence","And I go with the very last one"]
# encoded_input=tokenizer(batch_sentences,batch_of_second_sentences)
# print(encoded_input)