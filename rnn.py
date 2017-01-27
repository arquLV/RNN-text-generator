#!/usr/bin/env python3

import sys
import numpy
import random
import tensorflow as tf


# ./rnn.py obama train
# ./rnn.py obama test
savefile = 'model'
restore_for_training = False
if len(sys.argv) >= 3:
    dataset = sys.argv[1]
    data_file = "data/" + dataset + ".txt"
    mode = sys.argv[2]
    if len(sys.argv) == 4:
        savefile = sys.argv[3]
    if len(sys.argv) == 5 and sys.argv[4] == "-restore":
        restore_for_training = True

else:
    data_file = "data/binary.txt"
    mode = "train"

raw_data = open(data_file, 'r').read()  # Ieejas dati
raw_data = raw_data.lower()
alphabet = list(set(raw_data))        # Saraksts ar unikālajiem simboliem
alphabet_len = len(alphabet)


def vectorize_input(inp):
    # Katru simbolu datasetā aizstājam ar one-hot vektoru
    data = numpy.zeros([len(inp), alphabet_len])
    i = 0
    for ch in inp:
        one_hot = [0.0] * alphabet_len
        one_hot[alphabet.index(ch)] = 1.0
        data[i,:] = one_hot
        i += 1
    return data

data = vectorize_input(raw_data)

# print(data)

if mode == 'train':
    batch_size = 20
else:
    batch_size = 1
lstm_size = 128
lstm_layers = 2
num_steps = 50 # uz cik simboliem "atrullēt" RNN
learning_rate = 0.02

with tf.variable_scope("textgen"):
    # inputs = tf.placeholder(tf.float32, [batch_size, num_steps, alphabet_len])
    inputs = tf.placeholder(tf.float32, [None, None, alphabet_len])
    gold = tf.placeholder(tf.float32, [None, None, alphabet_len])
    gold_sh = tf.reshape(gold, [-1, alphabet_len])

    cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size)     
    lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_layers * [cell])
    initial_state = lstm.zero_state(batch_size, tf.float32)

    state = initial_state
    # for time_step in range(num_steps):
    #     enc_output, enc_state = lstm(inputs[:, time_step, :], state)
    #     state = enc_state

    # enc_output -- [batch_size, num_steps, output_size] 
    enc_output, enc_state = tf.nn.dynamic_rnn(lstm, inputs, initial_state=state, dtype=tf.float32)
    enc_output = tf.reshape(enc_output, [-1, lstm_size])

    weights = tf.Variable(tf.random_normal([lstm_size, alphabet_len], stddev=0.01))
    bias = tf.Variable(tf.random_normal([alphabet_len], stddev=0.01))

    # batch_size * alphabet_len
    # varbūtības katram alfabēta simbolam
    logits = tf.matmul(enc_output, weights) + bias
    probabilities = tf.nn.softmax(logits)       # softmax, lai varbūtības p0 + p1 + ... + pn = 1

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, gold_sh))
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cross_entropy)
            

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()


if mode == 'train':
    if restore_for_training == True:
        saver.restore(sess, savefile)

    batch = numpy.zeros([batch_size, num_steps, alphabet_len])
    gold_batch = numpy.zeros([batch_size, num_steps, alphabet_len])
    # gold_batch = numpy.zeros([batch_size, alphabet_len])

    iterations = 15000

    # Jāģenerē batchi...
    # vajadzīgas pēc skaita n=batch_size simbolu virknes garumā num_steps
    # Uzģenerējām kaut kādu random indeksu datu vektorā, lai aiz tā ir vēl vismaz num_steps simboli
    # Gold batchu ņemam ar nobīdi +1

    # Vispirms užģenerējam visus iespējamos sākumindeksus, no kuriem randomā ņemt
    # t.i. visi indeksi, izņemot pēdējos num_steps un vēl vienu, lai atstātu vietu +1 nobīdei
    possible_start_indices = range(data.shape[0] - num_steps - 1)
    for b in range(iterations):    
        start_indices = random.sample(possible_start_indices, batch_size)
        
        current_batch = 0
        for i in start_indices:
            batch[current_batch] = data[i:i+num_steps]
            gold_batch[current_batch] = data[i+1:i+num_steps+1]
            current_batch += 1

        feed = {inputs:batch, gold: gold_batch}

        cost, _ = sess.run([cross_entropy, train_step], feed_dict=feed)
        # print(output.shape)
        if b%50 == 0:
            print("Batch: ", b, "; cost: ", cost)

    saver.save(sess, savefile)
        
elif mode == 'test':

    if dataset == 'binary':
        root = '01'
    elif dataset == 'binary_two':
        root = '00'
    elif dataset == 'acbd':
        root = 'da'
    elif dataset == 'prog':
        root = '7 8 9'
    elif dataset == 'shakespeare':
        root = 'all:'
    else:
        root = 'the'

    saver.restore(sess, savefile)
    to_generate = 100

    for i in range(1,len(root)):
        out, state = sess.run([probabilities, enc_state], feed_dict={inputs: [vectorize_input(root[i])]})

    # print(out[0])
    gen = root
    for i in range(to_generate):
        ch = numpy.random.choice(range(alphabet_len), p=out[0])
        gen += alphabet[ch]
        out, state = sess.run([probabilities, enc_state], feed_dict={inputs: [vectorize_input(alphabet[ch])], initial_state: state})

    print(gen)