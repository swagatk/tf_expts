"""
Fine-tuning learned embeddings from Word2Vec
"""
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import nltk
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

np.random.seed(42)

input_file = './data/umich-sentiment-train.txt'
word2vec_model_path = '/home/swagat/Downloads/GoogleNews-vectors-negative300.bin.gz'
VOCAB_SIZE = 5000
EMBED_SIZE = 300
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 10

counter = collections.Counter()
fin = open(input_file, "rb")
maxlen = 0
for line in fin:
    _, sent = line.strip().split("t")
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        counter[word] += 1
fin.close()

word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_size = len(word2index)
index2word = {v:k for k, v in word2index.items()}

# Data preprocessing - padding / tokenizing
xs, ys = [], []
fin = open(input_file, "rb")
for line in fin:
    label, sent = line.strip().split("t")
    ys.append(int(label))
    words = [x.lower() for x in nltk.word_tokenize(sent)]
    wids = [word2index[word] for word in words]
    xs.append(wids)
fin.close()
X = keras.preprocessing.sequence.pad_sequences(xs, maxlen=maxlen)
Y = keras.utils.to_categorical(ys)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3,
                                                random_state=42)

# model
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, EMBED_SIZE, input_length=maxlen),
    keras.layers.SpatialDropout1D(keras.layers.Dropout(0.2)),
    keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS,
                        activation='relu'),
    keras.layers.GlobalMaxPool1D(),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer="adam",
              metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

# load word2vec model
word2vec = Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)
embedding_weights = np.zeros((vocab_size, EMBED_SIZE))
for word, index in word2index.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass

# Model
model = keras.models.Sequential([
    keras.layers.Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=maxlen,
                           weights=[embedding_weights]),
    keras.layers.SpatialDropout1D(keras.layers.Dropout(0.2)),
    keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS,
                        activation='relu'),
    keras.layers.GlobalMaxPool1D(),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE,
                    epochs=NUM_EPOCHS,
                    validation_data=(Xtest, ytest))
score = model.evaluate(Xtest, ytest)
print('Test Score: {:.3f}, accuracy: {:.3f}'.format(score[0], score[1]))