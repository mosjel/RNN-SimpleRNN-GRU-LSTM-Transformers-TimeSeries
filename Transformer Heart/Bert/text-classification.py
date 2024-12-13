# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="google-bert/bert-base-multilingual-cased")
s=pipe ("he is bad")
print(s)