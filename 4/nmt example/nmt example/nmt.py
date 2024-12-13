import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import re
import random
import string
import tensorflow as tf
from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

print(string.punctuation)
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
print(strip_chars)
exit

batch_size = 64
vocab_size = 15000
sequence_length = 20


def load_doc(path):
    
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")[:-1]

    return lines

def create_pairs(lines):
    text_pairs = []
    for line in lines:
        english, spanish = line.split("\t")
        spanish = "[start] " + spanish + " [end]"
        text_pairs.append((english, spanish))
    print("[INFO] printing sample data...")
    print(random.choice(text_pairs))
    return text_pairs

def split_data(text_pairs):

    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
    test_pairs = text_pairs[num_train_samples + num_val_samples:]

    print("[INFO] split dataset completed.")

    return train_pairs, val_pairs, test_pairs


def custom_standardization(input_string):

    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")

    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")


def tokenization(train_pairs):

    source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length)

    target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization)

    train_english_texts = [pair[0] for pair in train_pairs]
    train_spanish_texts = [pair[1] for pair in train_pairs]

    source_vectorization.adapt(train_english_texts)
    target_vectorization.adapt(train_spanish_texts)

    print("[INFO] data tokenized and convert to int numbers")

    return source_vectorization, target_vectorization

text_file =r"C:\Users\VAIO\Desktop\DSC\RNN\Week4\nmt example\nmt example\spa.txt"
lines = load_doc(text_file)
text_pairs = create_pairs(lines)

train_pairs, val_pairs, test_pairs = split_data(text_pairs)
source_vectorization, target_vectorization = tokenization(train_pairs)

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({

        "english": eng,
        "spanish": spa[:, :-1],
    }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f"targets.shape: {targets.shape}")

embed_dim = 256
latent_dim = 1024

def define_model():

    source = layers.Input(shape=(None,), dtype="int64", name="english")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)

    past_target = layers.Input(shape=(None,), dtype="int64", name="spanish")
    y = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)

    #Encoder
    encoded_source = layers.Bidirectional(layers.GRU(latent_dim), merge_mode="sum")(x)

    #Decoder
    y = layers.GRU(latent_dim, return_sequences=True)(y, initial_state=encoded_source)
    y = layers.TimeDistributed(layers.Dropout(0.5))(y)
    target_next_step = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"))(y)

    seq2seq_rnn = keras.Model([source, past_target], target_next_step)

    return seq2seq_rnn


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    '''
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation 
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    '''
    
    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()
    plt.show()


seq2seq_rnn = define_model()

seq2seq_rnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

seq2seq_history = seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

# Visualize the training and validation loss metrices.
plot_metric(seq2seq_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Visualize the training and validation accuracy metrices.
plot_metric(seq2seq_history, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy') 

