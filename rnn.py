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
    data_file = "data/" + sys.argv[1] + ".txt"
    mode = sys.argv[2]
    if len(sys.argv) == 4:
        savefile = sys.argv[3]
    if len(sys.argv) == 5 and sys.argv[4] == "-restore":
        restore_for_training = True

else:
    data_file = "data/obama.txt"
    mode = "train"

raw_data = open(data_file, 'r').read()  # Ieejas dati
raw_data = raw_data.lower()
alphabet = list(set(raw_data))        # Saraksts ar unikālajiem simboliem
alphabet_len = len(alphabet)

data = numpy.zeros([len(raw_data), alphabet_len])

# Katru simbolu datasetā aizstājam ar one-hot vektoru
i = 0
for ch in raw_data:
    one_hot = [0.0] * alphabet_len
    one_hot[alphabet.index(ch)] = 1.0
    data[i,:] = one_hot
    i += 1

batch_size = 32
# lstm_size = len(alphabet)
lstm_size = 128
lstm_layers = 2
num_steps = 50


# last_state = lstm.zero_state(batch_size, dtype=tf.float32)
last_state = numpy.zeros([lstm_layers*2*lstm_size])
with tf.variable_scope("textgen"):
    # encoder_inputs = tf.placeholder(tf.float32, [batch_size, num_steps, alphabet_len])
    encoder_inputs = tf.placeholder(tf.float32, [None, None, alphabet_len])
    cell = tf.nn.rnn_cell.LSTMCell(num_units=lstm_size)
    lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_layers * [cell])

    # lstm.state_size :: LSTMStateTuple
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard8/tf.nn.dynamic_rnn.md
    init_state = lstm.zero_state(batch_size, dtype=tf.float32)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(lstm, encoder_inputs, initial_state=init_state)

    # decoder_inputs = [encoder_outputs]
    # decoder_outputs, decoder_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, encoder_state, lstm)

    out_W = tf.Variable(tf.random_normal([lstm_size, alphabet_len], stddev=0.01))
    out_B = tf.Variable(tf.random_normal([alphabet_len], stddev=0.01))

    shaped_outputs = tf.reshape(encoder_outputs, [-1, lstm_size])
    raw_output = (tf.matmul(shaped_outputs, out_W) + out_B)

    batch_time_shape = tf.shape(encoder_outputs)
    output = tf.reshape(tf.nn.softmax(raw_output), (batch_time_shape[0], batch_time_shape[1], alphabet_len))

    gold = tf.placeholder(tf.float32, [None, None, alphabet_len])
    shaped_gold = tf.reshape(gold, [-1, alphabet_len])

    alpha = 0.03 # Learning rate
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(raw_output,shaped_gold))
    train_step = tf.train.AdadeltaOptimizer(alpha).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

if mode == 'train':
    iterations = 500
    if restore_for_training:
        saver.restore(sess, savefile)

    batch = numpy.zeros([batch_size, num_steps, alphabet_len])
    batch_gold = numpy.zeros([batch_size, num_steps, alphabet_len])

    pos_ids = range(data.shape[0] - num_steps - 1)

    for i in range(iterations):
        batch_id = random.sample(pos_ids, batch_size)

        for j in range(num_steps):
            idx_batch = [k+j for k in batch_id]
            idx_gold = [k+j+1 for k in batch_id]
            batch[:,j,:] = data[idx_batch,:]
            batch_gold[:,j,:] = data[idx_gold,:]

        init = numpy.zeros([batch.shape[0], lstm_layers*2*lstm_size])
        cost, _ = sess.run([cross_entropy, train_step], feed_dict={encoder_inputs:batch, gold:batch_gold})

        if i%50 == 0:
            print("Batch: ", i, " -- Cost: ", cost)
    

    saver.save(sess, savefile)
else: #test
    saver.restore(sess, savefile)

    root = "a "
    generate_len = 500
    for i in range(len(root)):
        rdata = numpy.zeros([1, alphabet_len])
        one_hot = [0.0]*alphabet_len
        one_hot[alphabet.index(root[i])] = 1.0
        rdata[0,:] = one_hot

        feed_data = [rdata]
        init = numpy.zeros([lstm_layers*2*lstm_size])
        out, next_state = sess.run([output, encoder_state], feed_dict={encoder_inputs:feed_data, init_state:init})
        last_state = next_state[0]

    generated = root
    for i in range(generate_len):
        ch = numpy.random.choice(range(alphabet_len), p=out)
        generated += vocab[ch]
        rdata = numpy.zeros([1, alphabet_len])
        one_hot = [0.0]*alphabet_len
        one_hot[alphabet.index(root[i])] = 1.0
        rdata[0,:] = one_hot

        feed_data = [rdata]
        init = last_state
        out, next_state = sess.run([output, encoder_state], feed_dict={encoder_inputs:feed_data, init_state:init})
        last_state = next_state[0]
    
    print(generated)

