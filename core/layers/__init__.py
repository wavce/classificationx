import tensorflow as tf
from .drop_block import DropBlock2D
from .normalizations import GroupNormalization
from .deformable_conv2d import DeformableConv2D
from .weight_standardization_conv2d import WSConv2D
# from .normalizations import SwitchableNormalization
from .normalizations import FilterResponseNormalization


def build_convolution(convolution, **kwargs):
    if convolution == "depthwise_conv2d":
        return tf.keras.layers.DepthwiseConv2D(**kwargs)
    elif convolution == "wsconv2d":
        return WSConv2D(**kwargs)
    elif convolution == "conv2d":
        return tf.keras.layers.Conv2D(**kwargs)
    elif convolution == "separable_conv2d":
        return tf.keras.layers.SeparableConv2D(**kwargs)
    elif convolution == "deformable_conv2d":
        return DeformableConv2D(**kwargs)
    else:
        raise TypeError("Could not interpret convolution function identifier: {}".format(repr(convolution)))


def build_normalization(**kwargs):
    normalization = kwargs.pop("normalization")
    if normalization == "group_norm":
        return GroupNormalization(**kwargs)
    elif normalization == "batch_norm":
        return tf.keras.layers.BatchNormalization(**kwargs)
    elif normalization == "switchable_norm":
        return SwitchableNormalization(**kwargs)
    elif normalization == "filter_response_norm":
        return FilterResponseNormalization(**kwargs)
    else:
        raise TypeError("Could not interpret normalization function identifier: {}".format(
            repr(normalization)))


def build_activation(**kwargs):
    activation = kwargs.pop("activation")
    if "leaky_relu" == activation:
        alpha = kwargs.pop("alpha")
        return tf.keras.layers.LeakyReLU(alpha=alpha)
    return tf.keras.layers.Activation(activation, **kwargs)


__all__ = [
    "WSConv2D",
    "DropBlock2D",
    "build_activation",
    "build_convolution",
    "GroupNormalization",
    "build_normalization",
    # "SwitchableNormalization",
]
