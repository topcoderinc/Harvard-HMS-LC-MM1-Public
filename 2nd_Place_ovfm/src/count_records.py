import glob
import tensorflow as tf
import sys


c = 0
for fn in glob.glob(sys.argv[1] + '*'):
  for record in tf.python_io.tf_record_iterator(fn, options = tf.python_io.TFRecordOptions(
          compression_type=tf.python_io.TFRecordCompressionType.ZLIB)):
     c += 1
print(c)
