from transformers import pipeline
classifier=pipeline("sentiment-analysis")
gg=classifier("He is rude")
print(gg)
"-------------"


# from transformers import pipeline
# ss=pipeline ("sentiment-analysis")
# f=ss(["i want to fuck", "I am horny"])
# score= [i["label"] for i in f]
# print(score)
# "--------"
# ss=pipeline("zero-shot-classification")
# f=ss("I am horny",candidate_labels=["economy","sport","love"])
# print(f)
# "-------------------"

# ss=pipeline("text-generation",model="distilgpt2")
# ss(" i want to fuck", max_length=30,num_return_sequences=2)
# "---------------------------"
# ss=pipeline("text-generation")
# ss(" i want to buy")
# "---------------------------"
# unmasker=pipeline("fill-mask")
# unmasker(" i want to kick her <mask> ",top_k=2)
# "---------------------------"
# ner=pipeline("ner",grouped_entities=True)
# ner("My name is Salyvan and i work in huggungface in brooklyn")
# "-------------------------------------"
# ss=pipeline("question-answering")
# ss(question='where i live',context="i am hamed and i am in tehran")
# "-------------------------------------"
# ss=pipeline("summarization")
# ss("""It’s pretty annoying that Keras doesn’t support Pickle to serialize its objects (Models). Yes, the Model structure is serializable (keras.models.model_from_json) and so are the weights (model.get_weights), and we can always use the built-in keras.models.save_model to store it as an hdf5 file, but all these won’t help when we want to store another object that references the model (like keras.callbacks.History), or use the %store magic of iPython notebook.

# After some frustration, I ended up with a patchy solution that does the work for me. It’s not the nicest thing, but works regardless of how you reference your Keras model. Basically, if an object has __getstate__ and __setstate__ methods, pickle will use them to serialize the object. The problem is that Keras Model doesn’t implement these. My patchy solution adds those methods after the class has been loaded:

# import types""")
# "---------------------------------"
# translator=pipeline("translation",model="Helsinki-NLP/opus-mt-fr-en")
# translator("Ce cours est produit par Hugging Face.")
