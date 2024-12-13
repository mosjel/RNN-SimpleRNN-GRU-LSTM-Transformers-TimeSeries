from keras.layers import TextVectorization
from keras.layers import Embedding
text_vectorization = TextVectorization(output_mode="int", output_sequence_length=15)

dataset = ["I write, erase, rewrite", "Erase again, and then", "A poppy blooms"]

test_data = ["write for natural language processing"]
test1_data=["erase the write and rewrite poppy blooms"]

text_vectorization.adapt(dataset)
s=(text_vectorization(test_data))
print(s)
s1=(text_vectorization(test1_data))
print(s1)
print(len(text_vectorization.get_vocabulary()))
out=Embedding(input_dim=12,output_dim=10)(s)
print(out)
out=Embedding(input_dim=12,output_dim=10)(s1)
print(out)

# ['', '[UNK]', 'erase', 'write', 'then', 'rewrite', 'poppy', 'i', 'blooms', 'and', 'again', 'a']