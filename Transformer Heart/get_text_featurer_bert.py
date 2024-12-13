# from transformers import BertTokenizer, TFBertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
# model = TFBertModel.from_pretrained("bert-base-multilingual-uncased")
# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)


#Pytorch Bert Model
import numpy as np
from numpy.linalg import norm
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained("bert-base-multilingual-uncased")
text = "سلام چطوری"
text1="حالت چطوره؟"
text2="اسم من حامده"

encoded_input = tokenizer(text, return_tensors='pt')
encoded_input1 = tokenizer(text1, return_tensors='pt')
encoded_input2 = tokenizer(text2, return_tensors='pt')

# dist=d=1-(np.dot(a,b)/(norm(a,axis=1)*norm(b)))
a = model(**encoded_input)[1].detach().numpy().squeeze()
b = model(**encoded_input1)[1].detach().numpy().squeeze()
c = model(**encoded_input2)[1].detach().numpy().squeeze()

print(a.shape)
print(type(a))
def distance(a,b):

    dist=d=(np.dot(a,b)/(norm(a)*norm(b)))

    return dist
print(distance(a,b))
print(distance(a,c))
print(distance(b,c))

# print (type(a))



