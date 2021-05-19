import tensorflow as tf


def get_drop_connect_rate(init_rate, block_num, total_blocks):
    """Get drop connect rate for the ith block."""
    if init_rate is not None:
        return init_rate * float(block_num) / total_blocks
    else:
        return None


class DropConnect(tf.keras.layers.Layer):
    """Apply drop connect.
        
        Args:
            rate: `float` drop connect rate.
        Returns:
            A output tensor, which should have the same shape as input.
  """
    def __init__(self, rate, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        
        self.keep_prob = rate if rate else 1.

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        super(DropConnect, self).build(input_shape)
    
    def drop_connect(self, inputs):
        batch_size = tf.shape(inputs)[0]
        random_tensor = self.keep_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.math.floor(random_tensor)
        output = tf.math.divide(inputs, self.keep_prob) * binary_tensor

        return output

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        return tf.cond(tf.logical_and(tf.cast(training, tf.bool), self.keep_prob < 1. and self.keep_prob > 0.),
                       lambda: self.drop_connect(inputs),
                       lambda: inputs)
    
    def get_config(self):
        config = {
            "rate": 1. - self.keep_prob
        }

        base_config = super(DropConnect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

