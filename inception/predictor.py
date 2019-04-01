"""Class for performing inference using DeepSolar paper model

Note: This class is formulated such that the graph set up to perform inference
is specific to the manner in which it was construcuted in the paper, lining up
with the weights that the authors made publicly available. To work with a
third-party implementation of the DeepSolar model, it would need to be reworked
a bit.
"""

import skimage.transform
import tensorflow as tf

from inception import inception_model as inception
from inception.slim import slim


class Predictor():
    """Predict using the DeepSolar paper model"""

    def __init__(self, dirpath_classification_checkpoint,
                 dirpath_segmentation_checkpoint=None, image_size=299,
                 num_classes=2):
        """Init

        :param dirpath_classification_checkpoint: directory path holding the
         tensorflow checkpoint for the classification head of the DeepSolar
         model
        :type dirpath_classification_checkpoint: str
        :param dirpath_segmentation_checkpoint: directory path holding the
         tensorflow checkpoint for the segmentation head of the DeepSolar
         model
        :type dirpath_segmentation_checkpoint: str
        :param image_size: square size of the input images used during model
         training; defaults to the 299 used in the paper
        :type image_size: int
        :param num_classes: number of classes in the target; defaults to the 2
         used in the paper
        """

        self.dirpath_classification_checkpoint = (
            dirpath_classification_checkpoint
        )
        self.dirpath_segmentation_checkpoint = (
            dirpath_segmentation_checkpoint
        )

        self.image_size = image_size
        self.num_classes = num_classes

        self.sess = tf.Session()
        self.input_placeholder, self.classify_op, self.segment_op = (
            self.load_inference_ops()
        )

    def _build_segment_op(self, feature_map_op):
        """Build the segmentation map op from the given `feature_map_op

        :param feature_map_op: feature map operation returned from the model
        :type feature_map_op: tensorflow.Tensor
        :return: operation that returns a segmentation map
        :rtype: tensorflow.Tensor
        """

        with tf.name_scope('conv_aux_1') as scope:
            kernel1 = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 288, 512], dtype=tf.float32, stddev=1e-4
                ), name='weights'
            )
            conv1 = tf.nn.conv2d(
                feature_map_op, kernel1, [1, 1, 1, 1], padding='SAME'
            )
            biases1 = tf.Variable(
                tf.constant(0.1, shape=[512], dtype=tf.float32),
                trainable=True, name='biases'
            )
            bias1 = tf.nn.bias_add(conv1, biases1)
            conv_aux1 = tf.nn.relu(bias1, name=scope)

        with tf.name_scope('conv_aux_2') as scope:
            kernel2 = tf.Variable(
                tf.truncated_normal(
                    [3, 3, 512, 512], dtype=tf.float32, stddev=1e-4
                ), name='weights'
            )
            conv2 = tf.nn.conv2d(
                conv_aux1, kernel2, [1, 1, 1, 1], padding='SAME'
            )
            biases2 = tf.Variable(
                tf.constant(0.1, shape=[512], dtype=tf.float32),
                trainable=True, name='biases'
            )
            bias2 = tf.nn.bias_add(conv2, biases2)
            conv_aux2 = tf.nn.relu(bias2, name=scope)

        random_normal_weights = tf.get_variable(
            name='W', shape=[512, 2],
            initializer=tf.random_normal_initializer(0., 0.01)
        )
        conv_aux2_resized = tf.image.resize_bilinear(conv_aux2, [100, 100])

        class_weights = tf.gather(tf.transpose(random_normal_weights), 1)
        class_weights = tf.reshape(class_weights, [-1, 512, 1])
        conv_aux2_resized = tf.reshape(conv_aux2_resized, [-1, 100 * 100, 512])
        class_activation_map_op = tf.matmul(conv_aux2_resized, class_weights)
        class_activation_map_op = tf.reshape(
            class_activation_map_op, [-1, 100, 100]
        )

        saver = tf.train.Saver(
            var_list=[
                random_normal_weights, kernel2, biases2, kernel1, biases1
            ]
        )
        checkpoint = tf.train.get_checkpoint_state(
            self.dirpath_segmentation_checkpoint
        )
        saver.restore(self.sess, checkpoint.model_checkpoint_path)

        return class_activation_map_op

    def classify(self, image):
        """Classify the provided image as having / not having solar panels

        :param image: pixel data to classify
        :type image: numpy.ndarray
        :return: probability that the image contains solar panels
        :rtype: float
        """

        score = self.sess.run(
            self.classify_op, feed_dict={self.input_placeholder: image}
        )
        return score[0][1]

    def load_inference_ops(self):
        """Build the ops that will perform inference

        :return: input placeholder and the operations to run to perform
         inference
        :rtype: 3-element tuple(tensorflow.Tensor)
        """

        num_channels = 3
        image_placeholder = tf.placeholder(
            tf.float32, (None, self.image_size, self.image_size, num_channels)
        )
        logits_op, _, feature_map_op = inception.inference(
            image_placeholder, self.num_classes
        )
        if self.dirpath_segmentation_checkpoint:
            segment_op = self._build_segment_op(feature_map_op)
        else:
            segment_op = None
        classification_prob_op = tf.nn.softmax(logits_op)

        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE
        )
        saver = tf.train.Saver(variables_to_restore)
        checkpoint = tf.train.get_checkpoint_state(
            self.dirpath_classification_checkpoint
        )
        saver.restore(self.sess, checkpoint.model_checkpoint_path)

        return image_placeholder, classification_prob_op, segment_op

    def segment(self, image, target_shape=None):
        """Segment solar panels on the provided image

        :param image: pixel data to segment
        :type image: numpy.ndarray
        :param target_shape: (height, width) to reshape the
         segmentation map to before returning it; defaults to the (100, 100)
         used in the DeepSolar model
        :type target_shape: tuple(int)
        :return: segmented pixel data
        :rtype: boolean numpy.ndarray
        """

        if self.segment_op is None:
            msg = (
                'self.segment_op is None, which means that a '
                'dirpath_segmentation_checkpoint was not passed into'
                'the __init__. To segment images, a '
                'dirpath_segmentation_checkpoint must be passed into the '
                '__init__.'
            )
            raise RuntimeError(msg)

        segmentation = self.sess.run(
            self.segment_op, feed_dict={self.input_placeholder: image}
        )[0]
        segmentation_rescaled = (
            (segmentation - segmentation.min()) /
            (segmentation.max() - segmentation.min())
        )

        if target_shape:
            segmentation_rescaled = skimage.transform.resize(
                segmentation_rescaled, target_shape
            )
        return segmentation_rescaled
