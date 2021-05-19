import math
import tensorflow as tf
import tensorflow_probability as tfp
from ..augmentations import Compose


class Dataset(object):
    def __init__(self, 
                 dataset_dir, 
                 training=True,
                 batch_size=32, 
                 augmentations=[],
                 mixup_alpha=0,
                 **kwargs):
        self.data_dir = dataset_dir
        self.training = training
        self.batch_size = batch_size
        self.mixup_alpha = mixup_alpha

        self.augment = Compose(augmentations)

    def mixup(self, images, labels):
        """Applies Mixup regularization to a batch of images and labels.

        [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
            Mixup: Beyond Empirical Risk Minimization.
            ICLR'18, https://arxiv.org/abs/1710.09412

        Args:
            batch_size: The input batch size for images and labels.
            alpha: Float that controls the strength of Mixup regularization.
            images: A batch of images of shape [batch_size, ...]
            labels: A batch of labels of shape [batch_size, num_classes]

        Returns:
            A tuple of (images, labels) with the same dimensions as the input with
            Mixup regularization applied.
        """
        mix_weight = tfp.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample([self.batch_size, 1])
        mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
        images_mix_weight = tf.reshape(mix_weight, [self.batch_size, 1, 1, 1])
        # Mixup on a single batch is implemented by taking a weighted sum with the same batch in reverse.
        images_mix = images * images_mix_weight + images[::-1] * (1. - images_mix_weight)
        labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)

        return images_mix, labels_mix

    def create_tfrecord(self, image_dir, image_info_file, output_dir, num_shards):
        raise NotImplementedError()

    def parser(self, serialized):
        raise NotImplementedError()

    def dataset(self):
        with tf.device("/cpu:0"):
            dataset = tf.data.TFRecordDataset(self.tf_record_sources)
            dataset = dataset.map(map_func=self.parser)
            if self.training:
                dataset = dataset.repeat() 
                dataset = dataset.shuffle(buffer_size=self.batch_size * 40)
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
           
            return dataset.prefetch(tf.data.experimental.AUTOTUNE)