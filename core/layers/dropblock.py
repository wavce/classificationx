import tensorflow as tf 


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, drop_rate, dropblock_size, data_format="channels_last", **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)

        self._keep_prob = 1. - drop_rate
        self._dropblock_size = dropblock_size
        self._data_format = data_format
    
    def build(self, input_shape):
        super(DropBlock2D, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def dropblock(self, inputs):
        """DropBlock: a regularization method for convolutional neural networks.
            
            DropBlock is a form of structured dropout, where units in a contiguous
            region of a feature map are dropped together. DropBlock works better than
            dropout on convolutional layers due to the fact that activation units in
            convolutional layers are spatially correlated.
            See https://arxiv.org/pdf/1810.12890.pdf for details.
            
            Args:
                net: `Tensor` input tensor.
                is_training: `bool` for whether the model is training.
                keep_prob: `float` or `Tensor` keep_prob parameter of DropBlock. "None"
                    means no DropBlock.
                dropblock_size: `int` size of blocks to be dropped by DropBlock.
                data_format: `str` either "channels_first" for `[batch, channels, height,
                    width]` or "channels_last for `[batch, height, width, channels]`.
            Returns:
                A version of input tensor with DropBlock applied.
            Raises:
                if width and height of the input tensor are not equal.
        """
        keep_prob, dropblock_size = self._keep_prob, self._dropblock_size
        data_format = self._data_format

        inp_shape = inputs.shape.as_list()
        if data_format == 'channels_last':
            width, height = inp_shape[1], inp_shape[2]
        else:
            width, height = inp_shape[2], inp_shape[3]
        
        tf.debugging.assert_equal(
            width, height, message="Input tensor with width!=height is not supported.")

        dropblock_size = min(dropblock_size, width)
        fdropblock_size = tf.cast(dropblock_size, tf.float32)
        fwidth = tf.cast(width, tf.float32)
        # seed_drop_rate is the gamma parameter of DropBlcok.
        seed_drop_rate = (
            (1.0 - keep_prob) * fwidth**2 / fdropblock_size**2 / 
            (fwidth - fdropblock_size + 1)**2)

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
        valid_block_center = tf.logical_and(
            tf.logical_and(w_i >= dropblock_size // 2, w_i < width - (dropblock_size - 1) // 2),
            tf.logical_and(h_i >= dropblock_size // 2, h_i < width - (dropblock_size - 1) // 2))

        valid_block_center = tf.expand_dims(valid_block_center, 0)
        valid_block_center = tf.expand_dims(
            valid_block_center, -1 if data_format == 'channels_last' else 0)

        randnoise = tf.random.uniform(tf.shape(inputs), dtype=tf.float32)
        block_pattern = (
            (1 - tf.cast(valid_block_center, dtype=tf.float32) + 
            tf.cast((1 - seed_drop_rate), dtype=tf.float32) + randnoise) >= 1)
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        if dropblock_size == width:
            block_pattern = tf.reduce_min(
                block_pattern,
                axis=[1, 2] if data_format == 'channels_last' else [2, 3],
                keepdims=True)
        else:
            block_pattern = -1. * tf.nn.max_pool(
                -block_pattern, 
                ksize=dropblock_size, 
                strides=1, 
                padding='SAME',
                data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

        percent_ones = (
            tf.cast(tf.reduce_sum((block_pattern)), tf.float32) / 
            tf.cast(tf.size(block_pattern), tf.float32))
        inputs /= (
            tf.cast(percent_ones, inputs.dtype) * 
            tf.cast(block_pattern, inputs.dtype))
        tf.print(inputs)

        return inputs

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        return tf.cond(
            tf.logical_and(
                tf.cast(training, tf.bool), 
                tf.cast(self._keep_prob > 0. and self._keep_prob < 1., tf.bool)),
            true_fn=lambda: self.dropblock(inputs),
            false_fn=lambda: inputs)

    def get_config(self):
        config = {
            "rate": 1. - self._keep_prob,
            "dropblock_size": self._dropblock_size,
            "data_format": self._data_format
        }

        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

