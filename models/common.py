import tensorflow as tf 
from core.layers import build_normalization
from core.layers import build_activation
from core.layers import DropBlock2D


class ConvNormActBlock(tf.keras.Model):
    """Conv2D-Norm-Activation block
    
    Args:
        filters(int): the filters of middle layer, i.e. conv2.
        kernel_size(int[tuple]): the kernel size.
        strides(int[tuple]): the strides.
        padding(str): one of `"valid"` or `"same"` (case-insensitive).
        groups(int): A positive integer specifying the number of groups in which the
            input is split along the channel axis. Each group is convolved separately
            with `filters / groups` filters. The output is the concatenation of all
            the `groups` results along the channel axis. Input channels and `filters`
            must both be divisible by `groups`.
        data_format(string): one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs. `channels_last` corresponds
            to inputs with shape `(batch_size, height, width, channels)` while
            `channels_first` corresponds to inputs with shape `(batch_size, channels,
            height, width)`. It defaults to the `image_data_format` value found in
            your Keras config file at `~/.keras/keras.json`. If you never set it, then
            it will be `channels_last`.
        dilation_rate(int[tuple]): an integer or tuple/list of 2 integers, specifying the
            dilation rate to use for dilated convolution. Can be a single integer to
            specify the same value for all spatial dimensions. Currently, specifying
            any `dilation_rate` value != 1 is incompatible with specifying any stride
            value != 1.
        kernel_initializer: Initializer for the `kernel` weights matrix (see `keras.initializers`).
        trainable(bool): if `True` the weights of this layer will be marked as
            trainable (and listed in `layer.trainable_weights`).
        normalization(dict): the normalization parameter dict.
        gamma_zeros(bool): bool, default False
            Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
        activation(dict): the activation paramerter dict.
        name(str): the block name.
    """
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=1, 
                 groups=1,
                 padding="same", 
                 data_format="channels_last", 
                 dilation_rate=1, 
                 kernel_initializer="he_normal",
                 trainable=True,
                 normalization=dict(normalization="batch_norm", axis=-1, trainable=True),
                 gamma_zeros=False,
                 activation=None,
                 dropblock=None,
                 name=None):
        super(ConvNormActBlock, self).__init__(name=name)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(strides, int):
            strides = (strides, strides)
        
        if strides != (1, 1):
            dilation_rate = (1, 1)

        if strides == (1, 1):
            padding == "same"
        else:
            p = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            self.pad = tf.keras.layers.ZeroPadding2D(p, data_format=data_format)
            padding = "valid"

        self.conv = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=kernel_size, 
                                           strides=strides, 
                                           padding=padding, 
                                           data_format=data_format, 
                                           dilation_rate=dilation_rate, 
                                           groups=groups,
                                           use_bias=False, 
                                           trainable=trainable,
                                           kernel_initializer=kernel_initializer,
                                           name="conv2d")
        if gamma_zeros:
            self.norm = build_normalization(**normalization, name=normalization["normalization"])
        else:
            self.norm = build_normalization(**normalization,
                                            gamma_initializer="zeros",
                                            name=normalization["normalization"])
        self.act = build_activation(**activation, name=activation["activation"]) if activation is not None else None
        self.dropblock = DropBlock2D(**dropblock, name="dropblock") if dropblock is not None else None

    def call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        x = self.norm(x, training=training)
        if self.act is not None:
            x = self.act(x)
        if self.dropblock is not None:
            x = self.dropblock(x, training=training)

        return x
    
    def fused_call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        
        return x


class DepthwiseConvNormActBlock(tf.keras.Model):
    """DepthwiseConv2D-Norm-Activation block"""
    def __init__(self, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding="same", 
                 data_format="channels_last", 
                 dilation_rate=1, 
                 kernel_initializer="he_normal",
                 trainable=True,
                 normalization=dict(axis=-1, trainable=True),
                 activation=None,
                 dropblock=None,
                 name=None):
        super(DepthwiseConvNormActBlock, self).__init__(name=name)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(strides, int):
            strides = (strides, strides)
        
        if strides != (1, 1):
            dilation_rate = (1, 1)
        
        if padding == "same":
            p = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            self.pad = tf.keras.layers.ZeroPadding2D(p, data_format=data_format)

        self.conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, 
                                                    strides=strides, 
                                                    padding="valid", 
                                                    data_format=data_format, 
                                                    dilation_rate=dilation_rate, 
                                                    use_bias=False, 
                                                    trainable=trainable,
                                                    kernel_initializer=kernel_initializer,
                                                    name="conv2d")
        self.norm = build_normalization(**normalization, name=normalization["normalization"])
        self.act = build_activation(**activation, name=activation["activation"]) if activation is not None else None
        self.dropblock = DropBlock2D(**dropblock, name="dropblock") if dropblock is not None else None

    def call(self, inputs, training=None):
        if hasattr(self, "pad"):
            x = self.pad(inputs)
        else:
            x = inputs
        x = self.conv(x)
        x = self.norm(x, training=training)
        if self.act is not None:
            x = self.act(x)
        
        if self.dropblock is not None:
            x = self.dropblock(x, training=training)

        return x


def fuse_conv_and_bn(conv, bn):
    fused_conv = tf.keras.layers.Conv2D(filters=conv.filters,
                                        kernel_size=conv.kernel_size,
                                        strides=conv.strides,
                                        padding=conv.padding,
                                        dilation_rate=conv.dilation_rate,
                                        use_bias=True)
    kernel_shape = tf.shape(conv.kernel)
    fused_conv(tf.random.uniform([1, 32, 32, kernel_shape[-2]]))
    
    # prepare kernel
    # conv_kernel = tf.transpose(tf.reshape(conv.kernel, [-1, kernel_shape[0]]))
    # bn_kernel = tf.linalg.diag(bn.gamma / tf.sqrt(bn.epsilon + bn.moving_variance))
    
    # kernel = tf.matmul(bn_kernel, conv_kernel)
    # kernel = tf.reshape(tf.transpose(conv_kernel), kernel_shape)
    bn_kernel = bn.gamma / tf.sqrt(bn.epsilon + bn.moving_variance)
    kernel = conv.kernel * bn_kernel
    fused_conv.kernel.assign(kernel)

    # prepare bias
    conv_bias = tf.zeros([kernel_shape[-1]], conv.kernel.dtype) if conv.bias is None else conv.bias
    bn_bias = bn.beta - bn.gamma * bn.moving_mean / tf.sqrt(bn.epsilon + bn.moving_variance)
    # bias = tf.reshape(tf.matmul(bn_kernel, tf.reshape(conv_bias, [-1, 1])), [kernel_shape[-1]]) + bn_bias
    bias = bn_kernel * conv_bias + bn_bias
    fused_conv.bias.assign(bias)

    return fused_conv


def fuse(model, building_block):
    print("Fusing layers ...")
    for l in model.layers:
        if isinstance(l, building_block):
            for ll in l.layers:
                if isinstance(ll, ConvNormActBlock) and isinstance(ll.norm, tf.keras.layers.BatchNormalization):
                    if ll.conv.groups == 1:
                        ll.conv = fuse_conv_and_bn(ll.conv, ll.norm)
                        ll.norm = None
                        ll.call = ll.fused_call
        if isinstance(l, ConvNormActBlock) and isinstance(l.norm, tf.keras.layers.BatchNormalization):
            l.conv = fuse_conv_and_bn(l.conv, l.norm)
            l.norm = None
            l.call = l.fused_call
