import sys
import tensorflow as tf
from tensorflow.nn import seq2seq

if(len(sys.argv) > 1):
    data_file = "data/" + sys.argv[1]
else:
    data_file = "data/obama.txt"

data = open(data_file, 'r').read()  # Ieejas dati
alphabet = list(set(data))        # Saraksts ar unikālajiem simboliem

chunk_size = 32     # Ņemam pa 32 simboliem

sess = tf.Session()

# cell = tf.nn.rnn_cell


# queue = tf.train.string_input_producer([data_file], num_epochs=None)
# reader = tf.WholeFileReader()

# _, record = reader.read(queue)
# features = tf.parse_single_example(record,
#     features={
#         'chunk': tf.FixedLenFeature([50], tf.string)
#     })

# chunk = features['chunk']    

# init = tf.initialize_all_variables()
# sess = tf.Session()
# tf.train.start_queue_runners(sess=sess)

# f1 = sess.run([chunk])
# f2 = sess.run([chunk    ])
# print(f1)
# print(f2)