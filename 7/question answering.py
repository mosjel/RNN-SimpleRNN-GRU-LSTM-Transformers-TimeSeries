from transformers import pipeline

ss=pipeline("question-answering")
dd=ss(question='where i live',context="i am hamed and i am in tehran")

print(dd)

