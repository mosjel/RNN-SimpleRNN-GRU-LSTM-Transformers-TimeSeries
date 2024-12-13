from keras.layers import TextVectorization

ds=['I write, erase, rewrite','Erase again, and then',' a poppy bloons']
print (len(ds))
text_vectorization=TextVectorization(output_mode="int")
text_vectorization.adapt(ds)
print(text_vectorization(["i go to school again"]))
print(text_vectorization.get_vocabulary())