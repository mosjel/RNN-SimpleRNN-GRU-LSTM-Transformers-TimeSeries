from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-uncased')
print(unmasker("Hello I'm a [MASK] model."))