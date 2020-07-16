import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = 'logs/word2vec_intro'

##################
# variable summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
################


digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
                     0: "PAD"}

sentences = []

for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
    sentences.append((" ".join([digit_to_word_map[r] for
                                    r in rand_odd_ints])))
    rand_even_ints = np.random.choice(range(2, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for
                                    r in rand_even_ints]))


# map words to indices
word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1
# inverse map
index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)

# create skip-grams pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1):
        word_context_pair = [[word2index_map[tokenized_sent[i-1]],
                              word2index_map[tokenized_sent[i+1]]],
                             word2index_map[tokenized_sent[i]]]
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][0]])
        skip_gram_pairs.append([word_context_pair[1],
                                word_context_pair[0][1]])

def get_skipgrams_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y


# Batch examples
x_batch, y_batch = get_skipgrams_batch(8)

# create placeholders
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


with tf.name_scope("embeddings"):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension],
                          -1.0, 1.0), name='embedding')
    # lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    variable_summaries(embed)

# Network weights
with tf.name_scope("rnn_weights"):
    with tf.name_scope('nce_wts'):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_dimension],
                                stddev=1.0/math.sqrt(embedding_dimension)))
        variable_summaries(nce_weights)
    with tf.name_scope('nce_b'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        variable_summaries(nce_biases)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, inputs=embed,
                       labels=train_labels, num_sampled=negative_samples,
                       num_classes=vocabulary_size))
    tf.summary.scalar('loss', loss)

with tf.name_scope('learning_rate'):
    # Learning rate decay
    global_step = tf.Variable(0, trainable=False)
    learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                              global_step=global_step,
                                              decay_steps=1000,
                                              decay_rate=0.95,
                                              staircase=True)
    tf.summary.scalar('learning_rate', learningRate)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

# Merge all summary ops
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    with open(os.path.join(LOG_DIR, 'metadata.tsv'), "w") as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name


    # Link embedding to its metadata file
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
    projector.visualize_embeddings(train_writer, config)

    tf.global_variables_initializer().run()
    for step in range(1000):
        x_batch, y_batch = get_skipgrams_batch(batch_size)
        summary, _ = sess.run([merged, train_step], feed_dict={
                                                    train_inputs: x_batch,
                                                    train_labels: y_batch})
        train_writer.add_summary(summary, step)
        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            loss_value = sess.run(loss, feed_dict={train_inputs: x_batch,
                                                   train_labels: y_batch})
            print("Loss at %d: %.5f" % (step, loss_value))

    # Normalize embeddings before use
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    normalized_embeddings_matrix = sess.run(normalized_embeddings)

    ref_word = normalized_embeddings_matrix[word2index_map["one"]]
    cosine_dists = np.dot(normalized_embeddings_matrix, ref_word)
    ff = np.argsort(cosine_dists)[::-1][1:10]
    for f in ff:
        print(index2word_map[f])
        print(cosine_dists[f])



