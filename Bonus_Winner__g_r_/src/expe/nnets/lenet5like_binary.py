# Create model for training / decoding

# Model for binary discrimation of scans: HAS/HASN'T tumor.

import tensorflow as tf

IMAGE_SIZE = 512
SIZE1 = IMAGE_SIZE-1  # After 1st conv
SIZE2 = SIZE1//2       # After 1st max pool
SIZE3 = SIZE2-1       # After 2nd conv
SIZE4 = SIZE3//4  -1     # After 2nd max pool = 63  . Why -1? Dunno
TARGET_SIZE = 1
PIXEL_DEPTH = 2200
NUM_CHANNELS = 3    # 3 slices stacked for features
SEED = 66478  # Set to None for random seed.


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    #if FLAGS.use_fp16:
    #  return tf.float16
    #else:
    return tf.float32

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.global_variables_initializer().run()}
conv1_weights = tf.Variable(
    tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter
                        stddev=0.1,
                        seed=SEED, dtype=data_type()))
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(tf.truncated_normal(
    [5, 5, 32, 64], stddev=0.1,
    seed=SEED, dtype=data_type()))
conv2_biases = tf.Variable(tf.constant(0.01, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([SIZE4 * SIZE4 * 64, 512],
                        stddev=0.03,
                        seed=SEED,
                        dtype=data_type()))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(tf.truncated_normal([512, TARGET_SIZE],
                                              stddev=0.03,
                                              seed=SEED,
                                              dtype=data_type()))
fc2_biases = tf.Variable(tf.constant(
    0.03, shape=[TARGET_SIZE], dtype=data_type()))


# We will replicate the model structure for the training sub-graph, as well
# as the evaluation sub-graphs, while sharing the trainable parameters.
def model(data_in, train=False):
    """The Model definition."""

    # 2D convolution, with 'VALID' padding (we don't mind the borders).
    # Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data_in,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID',
                          name='MaPoule1')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 4, 4, 1],
                          strides=[1, 4, 4, 1],
                          padding='VALID',
                          name='MaPoule2')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]],
        name='reshape/ovl')
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases, name='NotSoHidden')
    ## Add a 50% dropout during training only. Dropout also scales
    ## activations such that no rescaling is needed at evaluation time.
    #if train:
    #  hidden = tf.nn.dropout(hidden, 0.5, seed=SEED, name='DropOut')
    net = tf.matmul(hidden, fc2_weights, name="mymatmul") + fc2_biases
    output = tf.sigmoid(net)
    batch_size, _ = output.get_shape().as_list()  # may vary between train/valid/decoding
    return tf.reshape(output, [batch_size], "reshape2")

