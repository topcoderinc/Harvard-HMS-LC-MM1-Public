import numpy
import tensorflow as tf
from helpers.input_data import Dataset

from trainer import PIXEL_DEPTH, IMAGE_SIZE, TARGET_SIZE, NUM_CHANNELS, model

dataset = Dataset("/home/gerey/hms_lung/data/example_extracted_sample")

decode_data_node = tf.placeholder(
  tf.float32,
  shape=(1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
  name= "yg2_input_decoding")
#train_labels_node = tf.placeholder(
#  tf.float32,
#  shape=(1, TARGET_SIZE, TARGET_SIZE))

with tf.Session() as sess:

  batch_data, batch_labels = dataset.next_batch(1)
  # TODO: what if datatype()==tf.float16 ?
  batch_data = batch_data.astype(numpy.float32) / PIXEL_DEPTH - 0.5

  # This dictionary maps the batch data (as a numpy array) to the
  # node in the graph it should be fed to.
  feed_dict = {decode_data_node: batch_data }
               # train_labels_node: batch_labels}

  # Try to rebuild a saved graph doesn't seem to work
  #new_saver = tf.train.import_meta_graph('/home/gerey/hms_lung/models/no-weights-regul-0.meta')
  #forward_op = tf.get_collection('mon_decoder')[0]

  forward_op = model(decode_data_node)
  new_saver = tf.train.Saver()
  new_saver.restore(sess, '/home/gerey/hms_lung/models/no-weights-regul-2500')

  print(forward_op)
  x, = sess.run(forward_op, feed_dict=feed_dict)
  print(x)
