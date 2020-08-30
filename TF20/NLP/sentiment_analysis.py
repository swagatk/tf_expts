"""
Sentiment Analysis of Consumer Review
Tensorflow 2.0
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

### Avoid Cuda-NN load error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##########################

# Load the dataset
df = pd.read_csv('product_reviews_dataset.txt', encoding='ISO-8859-1')
print(df.columns)
print(df.head(10))
print(df.Sentiment.value_counts())

# text processing
def clean_reviews(text):
    # remove symbols, numbers, punctuations
    text = re.sub("[^a-zA-Z]", " ", str(text))
    return re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

df['Summary'] = df.Summary.apply(clean_reviews)
print(df.head(10))

# Prepare dataset for training
X = df.Summary
y = df.Sentiment
tokenizer = Tokenizer(num_words=10000, oov_token='xxxxxxx')
tokenizer.fit_on_texts(X)
X_dict = tokenizer.word_index
print('length of x_dict: {}'.format(len(X_dict)))
# print(X_dict.items())

X_seq = tokenizer.texts_to_sequences(X)
print(X_seq[:10])
X_padded_seq = pad_sequences(X_seq, padding='post', maxlen=100)
print('Shape of X_padded_seq: {}'.format(X_padded_seq.shape))

# convert pd data to numpy array for labels
y = np.array(y)
y = y.flatten()
print('Shape of labels: {}'.format(y.shape))


# Build the Deep Learning Model & train
text_model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(input_length=100, input_dim=10000, output_dim=50),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)
text_model.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])
text_model.summary()

# train
text_model.fit(X_padded_seq, y, epochs=10)


# Now there are 10,000 embeddings vectors of size 50
embeddings = text_model.layers[0]
print(embeddings.weights)
weights = embeddings.get_weights()[0]
print(weights.shape)

# Visualizing the word embeddings
# define reverse index for words
index_based_embedding = dict([(value, key) for (key, value) in X_dict.items()])


def decode_review(text):
    return ' '.join([index_based_embedding.get(i, '?') for i in text])


print(index_based_embedding[1])
print(weights[1])

vocab_size = 10000
vec = io.open('embedding_vectors_new.tsv', 'w', encoding='utf-8')
meta = io.open('metadata_new.tsv', 'w', encoding='utf-8')
for i in range(1, vocab_size):
    word = index_based_embedding[i]
    embedding_vec_values = weights[i]
    meta.write(word + "\n")
    vec.write('\t'.join([str(x) for x in embedding_vec_values]) + "\n")
meta.close()
vec.close()


