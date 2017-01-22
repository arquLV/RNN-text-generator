import tensorflow as tf

data_file = "data/obama.txt"

queue = tf.train.string_input_producer([data_file], num_epochs=None)
reader = tf.WholeFileReader()

_, record = reader.read(queue)
features = tf.parse_single_example(record,
    features={
        'chunk': tf.FixedLenFeature([50], tf.string)
    })

chunk = features['chunk']    

init = tf.initialize_all_variables()
sess = tf.Session()
tf.train.start_queue_runners(sess=sess)

f1 = sess.run([chunk])
f2 = sess.run([chunk    ])
print(f1)
print(f2)