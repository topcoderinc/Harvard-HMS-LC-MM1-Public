import tensorflow as tf
import tensorflow.contrib.layers as L
from abc import ABCMeta, abstractmethod
import utils
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import vgg


class Model(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, input_tensor):
        pass

    @abstractmethod
    def error(self, logits, targets):
        pass


class SegmentorModelLung(Model, metaclass=ABCMeta):
    def _get_weights(self, targets):
        weights = tf.zeros_like(targets, dtype=tf.float32)
        for i, w in enumerate(utils.LUNG_CLASS_WEIGHTS):
            mask = tf.to_float(tf.equal(targets, i))
            weights = weights + mask * w
        return weights

    def error(self, logits, targets):
        targets = tf.squeeze(targets)
        weights = self._get_weights(targets)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.to_int32(targets))
        nll =  nll * weights
        return tf.reduce_mean(nll)


class SegmentorModelRadiomics(Model, metaclass=ABCMeta):
    def _get_weights(self, targets):
        weights = tf.zeros_like(targets, dtype=tf.float32)
        for i, w in enumerate(utils.CLASS_WEIGHTS):
            mask = tf.to_float(tf.equal(targets, i))
            weights = weights + mask * w
        return weights

    def error(self, logits, targets):
        targets = tf.squeeze(targets)
        weights = self._get_weights(targets)
        nll = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.to_int32(targets))
        nll =  nll * weights
        return tf.reduce_mean(nll)


class MultiLabelClassifier(Model, metaclass=ABCMeta):
    def error(self, logits, targets):
        nll = tf.nn.sigmoid_cross_entropy_with_logits(
            logits, tf.to_float(targets))
        nll = tf.reduce_mean(nll, axis=0)
        # return tf.reduce_mean(nll * tf.constant(utils.CLASS_WEIGHTS, dtype=tf.float32))
        return tf.reduce_mean(nll)


class Encoder(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, input_tensor, is_training):
        pass


class ConvolutionalEncoderFirst256(Encoder):
    def forward(self, input_tensor, is_training):
        h = L.convolution2d(input_tensor, 16, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d(h, 16, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d(h, 32, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d(h, 64, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d(h, 128, [5, 5], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)
        return h


class ConvolutionalMultiLabelFirst256(MultiLabelClassifier):
    def forward(self, feature_map, is_training):

        h = L.convolution2d(feature_map, 256, [7, 7], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)
        h = L.max_pool2d(h, [2, 2], [2, 2])

        batch_size, height, width, depth = h.get_shape()

        h = tf.reshape(h, tf.stack([-1, height * width * depth]))
        logits = L.fully_connected(h, len(STRUCTURE_CLASS) + 1, activation_fn=None)

        return logits


class ConvolutionalSegmentationFirst256(SegmentorModelRadiomics):
    def __init__(self):
        self.encoder = ConvolutionalEncoderFirst256()

    def _decode(self, feature_map, is_training):
        h = L.convolution2d_transpose(feature_map, 128, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=tf.nn.relu)

        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)
        return h

    def forward(self, images, is_training):
        feature_map = self.encoder.forward(images, is_training)
        return self._decode(feature_map, is_training)


class ConvolutionalSegmentationModel(SegmentorModelLung):
    def forward(self, input_tensor, is_training):
        dropout_value = 0.5
        h = L.convolution2d(input_tensor, 16, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 32, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 64, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 128, [5, 5], activation_fn=tf.nn.relu)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 256, [5, 5], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 128, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 16, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, len(utils.LUNG_CLASSES) + 1, [1, 1], [1, 1], activation_fn=None)

        return h


class ConvolutionalSegmentationModelRadiomics(SegmentorModelRadiomics):
    def forward(self, input_tensor, is_training):
        dropout_value = 0.5

        h = L.convolution2d(input_tensor, 16, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 32, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 64, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.max_pool2d(h, [2, 2], [2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, 128, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 128, [5, 5], [2, 2], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], activation_fn=None)
        h = L.batch_norm(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)

        return h


class ConvolutionalSegmentationModel3D(SegmentorModelRadiomics):
    def forward(self, input_tensor, is_training):
        dropout_value = 0.5
        input_tensor = tf.expand_dims(input_tensor, -1)

        h = tf.layers.conv3d(input_tensor, 16, [5, 5, 3], padding="same")
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling3d(h, [2, 2, 2], [2, 2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)
        print(h)

        h = tf.layers.conv3d(h, 32, [5, 5, 2], padding="same")
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling3d(h, [2, 2, 1], [2, 2, 1])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)
        print(h)

        h = tf.layers.conv3d(h, 64, [5, 5, 2], padding="same")
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = tf.layers.max_pooling3d(h, [2, 2, 2], [2, 2, 2])
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)
        print(h)

        h = tf.layers.conv3d(h, 128, [5, 5, 1], padding="same")
        h = tf.layers.batch_normalization(h)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        print(h)
        h = tf.squeeze(h, 3)
        print(h)

        h = L.convolution2d_transpose(h, 128, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=tf.nn.relu)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)


        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)

        return h


class VGGModel(SegmentorModelRadiomics):
    def forward(self, input_tensor, is_training):
        dropout_value = 0.5

        print("Is training:", is_training)
        input_tensor = tf.image.resize_images(input_tensor, [224, 224])
        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, endpoint = vgg.vgg_19(input_tensor, is_training=is_training)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")

        h = L.convolution2d_transpose(h, 128, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)
        print(h)

        return h


class VGGModel(SegmentorModelRadiomics):
    def forward(self, input_tensor, is_training):
        dropout_value = 0.5

        input_tensor = tf.image.resize_images(input_tensor, [224, 224])

        print("Is training:", is_training)

        with slim.arg_scope(vgg.vgg_arg_scope()):
            h, end_points = vgg.vgg_19(input_tensor, is_training=is_training)

        h = tf.pad(end_points['vgg_19/conv4'], [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
        print(h)

        h = L.convolution2d_transpose(h, 128, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 64, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=dropout_value, is_training=is_training)

        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)

        return h



class ResnetV2Segmentation(SegmentorModelRadiomics):
    def forward(self, input_tensor, is_training):
        # inputs has shape [batch, 513, 513, 3]
        input_tensor = tf.image.resize_images(input_tensor, [512, 512])
        with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training)):
           net, end_points = resnet_v1.resnet_v1_101(input_tensor,
                                           None,
                                           global_pool=False,
                                           output_stride=16)
           print(net.get_shape())

        h = L.convolution2d_transpose(net, 64, [5, 5], [4, 4], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        h = L.convolution2d_transpose(h, 32, [5, 5], [2, 2], activation_fn=None)
        h = tf.nn.relu(h)
        h = L.dropout(h, keep_prob=0.5, is_training=is_training)

        print(h)

        h = L.convolution2d(h, len(utils.STRUCTURE_CLASS) + 1, [1, 1], [1, 1], activation_fn=None)
        print(h)
        return h
