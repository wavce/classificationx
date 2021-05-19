import tensorflow as tf 
from core.layers.normalizations import build_normalization


def _create_broadcast_shape(input_shape, axis):
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]
    return broadcast_shape
    

@tf.custom_gradient
def _mish(x):
    x1 = tf.nn.tanh(tf.nn.softplus(x))
    
    def _grad(dy):
        dx = x1 + x * tf.nn.sigmoid(x) * (1 - x1 * x1)
        return dx * dy
    
    return x * x1, _grad


class FReLU(tf.keras.layers.Layer):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)


class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        # x = inputs * (tf.nn.tanh(tf.nn.softplus(inputs)))
        
        return _mish(inputs)

    def get_config(self):
        return super().get_config()


class ACONC(tf.keras.layers.Layer):
    r""" ACON activation (activate or not).
        (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x,  beta is a learnable parameter
        according to "Activate or Not: Learning Customized Activation" 
        <https://arxiv.org/pdf/2009.04759.pdf>.
    """
    def __init__(self, 
                 width, 
                 r=16, 
                 data_format="channels_last", 
                 **kwargs):
        super().__init__(**kwargs)

        self.width = width
        self.data_format = data_format
        self.axis = -1 if data_format == "channels_last" else 1
    
    def build(self, input_shape):
        self.p1 = self.add_weight(
            name="p1",
            shape=[self.width],
            dtype=self.dtype,
            trainable=True)
        self.p2 = self.add_weight(
            name="p2",
            shape=[self.width],
            dtype=self.dtype,
            trainable=True)
        self.beta = self.add_weight(
            name="beta",
            shape=[self.width],
            dtype=self.dtype,
            trainable=True,
            initializer=tf.keras.initializers.Ones())
        
        self.built = True
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        broadcast_shape = _create_broadcast_shape(inputs.shape.as_list(), self.axis)
        p1 = tf.reshape(self.p1, broadcast_shape)
        p2 = tf.reshape(self.p2, broadcast_shape)
        beta = tf.reshape(self.beta, broadcast_shape)
        x = inputs
        x = (p1 * x - p2 * x) * tf.nn.sigmoid(beta * (p1 * x - p2 * x)) + p2 * x

        return x
    
    def get_config(self):
        config = {
            "width": self.width,
            "data_format": self.data_format
        }
        config.update(super().get_config())
        return config


class MetaACON(tf.keras.layers.Layer):
    def __init__(self, 
                 width, 
                 r=16, 
                 data_format="channels_last", 
                 normalization=dict(
                     normalization="batch_norm", 
                     momentum=0.9, 
                     epsilon=1e-5, 
                     axis=-1, 
                     trainable=True), 
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.r = r
        self.data_format = data_format

        self.normalization = normalization
        self.normalization["trainable"] = trainable
        self.axis = -1 if data_format == "channels_last" else 1
        self.normalization["axis"] = self.axis

    def build(self, input_shape):
        self.fc1 = tf.keras.layers.Conv2D(
            filters=max(self.r, self.width // self.r),
            kernel_size=1,
            padding="same",
            data_format=self.data_format,
            use_bias=True,
            name="fc1")
        self.norm1 = build_normalization(
            name="norm1", 
            **self.normalization)
        self.fc2 = tf.keras.layers.Conv2D(
            filters=self.width,
            kernel_size=1,
            padding="same",
            data_format=self.data_format,
            use_bias=True,
            name="fc2")
        self.norm2 = build_normalization(
            name="norm2", 
            **self.normalization)
        
        self.p1 = self.add_weight(
            name="p1",
            shape=[self.width],
            dtype=self.dtype,
            trainable=True)
        self.p2 = self.add_weight(
            name="p2",
            shape=[self.width],
            dtype=self.dtype,
            trainable=True)
        
        self.built = True
    
    def call(self, inputs, training=None):
        broadcast_shape = _create_broadcast_shape(inputs.shape.as_list(), self.axis)
        p1 = tf.reshape(self.p1, broadcast_shape)
        p2 = tf.reshape(self.p2, broadcast_shape)

        beta = tf.reduce_mean(
            inputs, 
            axis=[1, 2] if self.axis == -1 else [2, 3],
            keepdims=True)
        beta = self.fc1(beta)
        beta = self.norm1(beta, training=training)
        beta = self.fc2(beta)
        beta = self.norm2(beta, training=training)
        beta = tf.nn.sigmoid(beta)

        x = inputs
        x = (p1 * x - p2 * x) * tf.nn.sigmoid(beta * (p1 * x - p2 * x)) + p2 * x

        return x
    
    def get_config(self):
        config = {
            "width": self.width,
            "data_format": self.data_format,
            "r": self.r
        }
        config.update(super().get_config())

        return config


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.nn.relu6(inputs + 3.0) / 6.0
    
    def get_config(self):
        return super().get_config()


def build_activation(**kwargs):
    activation = kwargs.pop("activation").lower()
    if activation.lower() == "leaky_relu":
        alpha = kwargs.pop("alpha")
        return tf.keras.layers.LeakyReLU(alpha=alpha, **kwargs)
    
    if activation == "prelu":
        return tf.keras.layers.PReLU(
            alpha_initializer=tf.keras.initializers.Constant(0.25), 
            **kwargs)
    
    if activation == "aconc":
        return ACONC(**kwargs)
    
    if activation == "meta_acon" or activation == "metaacon":
        return MetaACON(**kwargs)
    
    if activation == "relu6":
        return tf.keras.layers.ReLU(max_value=6, **kwargs)
    
    if activation == "hard_sigmoid" or activation == "hardsigmoid":
        return HardSigmoid(**kwargs)

    return tf.keras.layers.Activation(activation, **kwargs)
