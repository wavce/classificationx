import tensorflow  as tf 
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from core.layers import build_activation


import sys
sys.setrecursionlimit(100000)

class RSoftMax(tf.keras.layers.Layer):
    def __init__(self, radix, cardinality, **kwargs):
        super(RSoftMax, self).__init__(**kwargs)

        self.radix = radix
        self.cardinality = cardinality
    
    def call(self, inputs):
        shape = tf.shape(inputs)
        b = shape[0]
        
        if self.radix > 1:
            x = tf.reshape(inputs, [b, self.cardinality, self.radix, -1])
            x = tf.transpose(x, [0, 2, 1, 3])
            x = tf.nn.softmax(x, axis=1)
            x = tf.reshape(x, shape)
        else:
            x = tf.nn.sigmoid(inputs)
        
        return x


def SplAtConv2d(inputs,
                filters,
                in_filters,
                kernel_size,
                strides=1,
                dilation_rate=1,
                cardinality=1,
                radix=2,
                reduction_factor=4,
                data_format="channels_last",
                trainable=True,
                normalization=dict(
                    normalization="batch_norm", 
                    momentum=0.9, 
                    epsilon=1e-5, 
                    axis=-1, 
                    trainable=True),
                dropblock=dict(dropdropblock_size=7, rate=0.1), 
                activation=dict(activation="relu"),
                name="splat_conv2d"):
    channel_axis = -1 if (data_format == "channels_last" or data_format is None) else 1
    x = ConvNormActBlock(
        filters=filters * radix,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        groups=cardinality * radix,
        trainable=trainable,
        data_format=data_format,
        normalization=normalization,
        dropblock=dropblock,
        activation=activation,
        name=name + "/conv")(inputs)
    # Split 
    if radix > 1:
        splitted = tf.keras.layers.Lambda(
            lambda inp: tf.split(inp, radix, axis=channel_axis),
            name=name + "/split_radix0")(x)
        gap = tf.keras.layers.Add(name=name + "/add0")(splitted)
    else:
        gap = x
    
    avg_axis = [1, 2] if channel_axis == -1 else [2, 3]
    gap = tf.keras.layers.Lambda(
        lambda inp: tf.reduce_mean(inp, avg_axis, keepdims=True),
        name=name + "/global_avgpool")(gap)

    inter_filters = max(in_filters * radix // reduction_factor, 32)
    gap = ConvNormActBlock(
        filters=inter_filters,
        kernel_size=(1, 1),
        trainable=trainable,
        groups=cardinality,
        use_bias=True,
        normalization=normalization,
        activation=activation,
        name=name + "/fc1")(gap)
    gap = ConvNormActBlock(
        filters=filters * radix,
        kernel_size=(1, 1),
        trainable=trainable,
        groups=cardinality,
        normalization=None,
        use_bias=True,
        activation=None,
        name=name + "/fc2")(gap)
    attn = RSoftMax(radix, cardinality, name=name + "/rsoftmax")(gap)

    if radix > 1:
        attns = tf.keras.layers.Lambda(
            lambda inp: tf.split(inp, radix, axis=channel_axis),
            name=name + "/split_radix1")(attn)
  
        outputs = [
            tf.keras.layers.Multiply(
                name=name + "/multiply%d" % i)([a, s]) 
            for i, (a, s) in enumerate(zip(attns, splitted))]
        outputs = tf.keras.layers.Add(name=name + "/add2")(outputs)
    else:
        outputs = tf.keras.layers.Multiply(name=name + "/multiply")[attn, x]
    
    return outputs


def bottleneck(inputs, 
               filters,
               in_filters,
               radix=1,
               cardinality=1, 
               bottleneck_width=64, 
               strides=1, 
               dilation_rate=1,
               data_format="channels_last",
               dropblock=None,
               normalization=dict(
                   normalization="batch_norm", 
                   momentum=0.9, 
                   epsilon=1e-5, 
                   axis=-1, 
                   trainable=True),
               activation=dict(activation="relu"),
               avg_down=False,
               last_gamma=False,
               avd=False,
               avd_first=False,
               is_first=False,
               trainable=True,
               name="bottleneck"):
    avd = avd and (strides > 1 or is_first)
    act = activation["activation"]
    group_width = int(filters * (bottleneck_width / 64.)) * cardinality
    x = ConvNormActBlock(
        filters=group_width, 
        kernel_size=(1, 1), 
        data_format=data_format,
        trainable=trainable,
        normalization=normalization,
        activation=activation,
        dropblock=dropblock,
        name=name + "/conv1")(inputs)

    if avd and avd_first:
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.AvgPool2D(
            pool_size=(3, 3), 
            strides=strides, 
            padding="valid", 
            data_format=data_format,
            name=name + "/avd_layer")(x)
    
    if radix > 1:
        x = SplAtConv2d(
            x,
            filters=group_width,
            in_filters=group_width,
            kernel_size=(3, 3),
            strides=1 if avd else strides,
            dilation_rate=dilation_rate,
            cardinality=cardinality,
            radix=radix,
            reduction_factor=4,
            data_format=data_format,
            normalization=normalization,
            dropblock=dropblock, 
            activation=activation,
            trainable=trainable,
            name=name + "/conv2")
    else:
        x = ConvNormActBlock(
            filters=group_width, 
            kernel_size=(3, 3), 
            strides=1 if avd else strides,
            padding="same",
            use_bias=False,
            groups=cardinality,
            trainable=trainable,
            data_format=data_format,
            normalization=normalization,
            dropblock=dropblock,
            activation=activation,
            name=name + "/conv2")(x)
    
    if avd and not avd_first:
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.AvgPool2D(
            pool_size=(3, 3), 
            strides=strides, 
            padding="valid", 
            data_format=data_format,
            name=name + "/avd_layer")(x)
    
    x = ConvNormActBlock(
        filters=filters * 4, 
        kernel_size=(1, 1), 
        trainable=trainable,
        data_format=data_format,
        normalization=normalization,
        dropblock=dropblock,
        activation=None,
        gamma_zeros=last_gamma,
        name=name + "/conv3")(x)

    shortcut = inputs
    if strides != 1 or filters * 4 != in_filters:
        if avg_down:
            if dilation_rate == 1:
                shortcut = tf.keras.layers.AvgPool2D(
                    pool_size=strides, 
                    strides=strides, 
                    padding="same", 
                    data_format=data_format,
                    name=name + "/downsample/avgpool")(shortcut)
            else:
                shortcut = tf.keras.layers.AvgPool2D(
                    pool_size=1, 
                    strides=1, 
                    padding="same", 
                    data_format=data_format,
                    name=name + "/downsample/avgpool")(shortcut)
            strides = 1
    
        shortcut = ConvNormActBlock(
            filters=filters * 4, 
            kernel_size=1, 
            strides=strides,
            use_bias=False,
            padding="same",
            trainable=trainable,
            data_format=data_format,
            normalization=normalization,
            dropblock=dropblock,
            activation=None,
            name=name + "/downsample")(shortcut)
    x = tf.keras.layers.Add(name=name + "/sum")([x, shortcut])
    x = build_activation(**activation, name=name + "/%s3" % act)(x)

    return x


class ResNeSt(Model):
    def __init__(self,
                 name,
                 block_fn,
                 num_blocks,
                 deep_stem=False,
                 stem_width=32,
                 radix=2,
                 cardinality=1,
                 bottleneck_width=64,
                 normalization=dict(),
                 last_gamma=False,
                 activation=dict(),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=None,
                 input_tensor=None,
                 dropblock=None,
                 avg_down=False,
                 avd=False,
                 avd_first=False,
                 num_classes=1000,
                 drop_rate=0.5):
        super(ResNeSt, self).__init__(
            name, 
            normalization=normalization, 
            activation=activation, 
            output_indices=output_indices, 
            strides=strides, 
            dilation_rates=dilation_rates, 
            frozen_stages=frozen_stages, 
            input_shape=input_shape, 
            input_tensor=input_tensor, 
            dropblock=dropblock, 
            num_classes=num_classes, 
            drop_rate=drop_rate)
        self.block_fn = block_fn
        self.num_blocks = num_blocks
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.stem_width = stem_width 
        self.deep_stem = deep_stem
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first
        self.avg_down = avg_down
        self.last_gamma = last_gamma
    
    def build_model(self):
        def _norm(inp):
            mean = tf.constant([0.485, 0.456, 0.406], inp.dtype, [1, 1, 1, 3]) * 255.
            std = 1. / (tf.constant([0.229, 0.224, 0.225], inp.dtype, [1, 1, 1, 3]) * 255.)
            inp = (inp - mean) * std  
           
            return inp
        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input) 

        if not self.deep_stem:
            x = ConvNormActBlock(
                filters=64, 
                kernel_size=7, 
                strides=2, 
                data_format=self.data_format, 
                normalization=self.normalization,
                activation=self.activation,
                dropblock=self.dropblock,
                name="conv1")(x)
        else:
            x = ConvNormActBlock(
                filters=self.stem_width, 
                kernel_size=3, 
                strides=2, 
                data_format=self.data_format, 
                normalization=self.normalization,
                activation=self.activation,
                dropblock=self.dropblock,
                name="conv1/1")(x)
            x = ConvNormActBlock(
                filters=self.stem_width, 
                kernel_size=3, 
                strides=1, 
                data_format=self.data_format, 
                normalization=self.normalization,
                activation=self.activation,
                dropblock=self.dropblock,
                name="conv1/2")(x)
            x = ConvNormActBlock(
                filters=self.stem_width * 2, 
                kernel_size=3, 
                strides=1, 
                data_format=self.data_format, 
                normalization=self.normalization,
                activation=self.activation,
                dropblock=self.dropblock,
                name="conv1/3")(x)
            
        outputs = [x]
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), 
            strides=self.strides[1], 
            padding="valid", 
            data_format=self.data_format,
            name="maxpool0")(x)
        x = self._make_layer(
            x, 64, 64, self.num_blocks[0], 1, self.dilation_rates[1], 
            2 not in self.frozen_stages, False, name="layer1")
        outputs.append(x)
        x = self._make_layer(
            x, 128, 256, self.num_blocks[1], self.strides[2], 
            self.dilation_rates[2], 3 not in self.frozen_stages, name="layer2")
        outputs.append(x)
        x = self._make_layer(
            x, 256, 512, self.num_blocks[2], self.strides[3], 
            self.dilation_rates[3], 4 not in self.frozen_stages, name="layer3")
        outputs.append(x)
        x = self._make_layer(
            x, 512, 1024, self.num_blocks[3], self.strides[4], 
            self.dilation_rates[4], 5 not in self.frozen_stages, name="layer4")
        outputs.append(x)

        if -1 not in self.output_indices:
            return (outputs[i-1] for i in self.output_indices)

        x = tf.keras.layers.GlobalAvgPool2D(
            data_format=self.data_format, 
            name="gloabl_avgpool")(x)
        if self.drop_rate and self.drop_rate > 0.:
            x = tf.keras.layers.Dropout(rate=self.drop_rate, name="drop")(x)
        
        outputs = tf.keras.layers.Dense(units=self.num_classes, name="fc")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def _make_layer(self, inputs, filters, in_filters, num_block, 
                    strides=1, dilation_rate=1, trainable=True, is_first=True, name="layer"):
        if dilation_rate == 1 or dilation_rate == 2:
            x = self.block_fn(
                inputs, 
                filters,
                in_filters,
                radix=self.radix,
                cardinality=self.cardinality, 
                bottleneck_width=self.bottleneck_width, 
                strides=strides, 
                dilation_rate=1,
                data_format=self.data_format,
                dropblock=self.dropblock,
                normalization=self.normalization,
                activation=self.activation,
                avg_down=self.avg_down,
                last_gamma=self.last_gamma,
                avd=self.avd,
                avd_first=self.avd_first,
                is_first=is_first,
                trainable=True,
                name=name + "/0")
        elif dilation_rate == 4:
            x = self.block_fn(
                inputs, 
                filters,
                in_filters,
                radix=self.radix,
                cardinality=self.cardinality, 
                bottleneck_width=self.bottleneck_width, 
                strides=strides, 
                dilation_rate=2,
                data_format=self.data_format,
                dropblock=self.dropblock,
                normalization=self.normalization,
                activation=self.activation,
                avg_down=self.avg_down,
                last_gamma=self.last_gamma,
                avd=self.avd,
                avd_first=self.avd_first,
                is_first=is_first,
                trainable=True,
                name=name + "/0")
        else:
            raise ValueError("Unknown dilation size: {}".format(dilation_rate))
        
        in_filters = filters * 4
        for i in range(1, num_block):
            x = self.block_fn(
                x, 
                filters,
                in_filters,
                radix=self.radix,
                cardinality=self.cardinality, 
                bottleneck_width=self.bottleneck_width, 
                strides=1, 
                dilation_rate=dilation_rate,
                data_format=self.data_format,
                dropblock=self.dropblock,
                normalization=self.normalization,
                activation=self.activation,
                avg_down=self.avg_down,
                last_gamma=self.last_gamma,
                avd=self.avd,
                avd_first=self.avd_first,
                trainable=True,
                name=name + "/%d" % i)
        return x


@MODELS.register("ResNeSt50")
def ResNeSt50(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              output_indices=(-1, ), 
              strides=(2, 2, 2, 2, 2), 
              dilation_rates=(1, 1, 1, 1, 1), 
              frozen_stages=(-1, ), 
              input_shape=None, 
              input_tensor=None, 
              dropblock=dict(dropblock_size=7, drop_rate=0.1), 
              last_gamma=False,
              num_classes=1000,
              drop_rate=0.5,
              **kwargs):
    return ResNeSt(
        name="resnest50",
        block_fn=bottleneck,
        num_blocks=[3, 4, 6, 3],
        deep_stem=True,
        stem_width=32,
        radix=2,
        cardinality=1,
        bottleneck_width=64,
        normalization=normalization,
        last_gamma=last_gamma,
        activation=activation,
        output_indices=output_indices,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        input_shape=input_shape,
        input_tensor=input_tensor,
        dropblock=dropblock,
        avg_down=True,
        avd=True,
        avd_first=False,
        num_classes=num_classes,
        drop_rate=drop_rate).build_model()


@MODELS.register("ResNeSt101")
def ResNeSt101(normalization=dict(
                    normalization="batch_norm", 
                    momentum=0.9, 
                    epsilon=1e-5, 
                    axis=-1, 
                    trainable=True),
               activation=dict(activation="relu"),
               output_indices=(-1, ), 
               strides=(2, 2, 2, 2, 2), 
               dilation_rates=(1, 1, 1, 1, 1), 
               frozen_stages=(-1, ), 
               input_shape=None, 
               input_tensor=None, 
               dropblock=dict(dropblock_size=7, drop_rate=0.1), 
               last_gamma=False,
               num_classes=1000,
               drop_rate=0.5,
               **kwargs):
        return ResNeSt(
            name="resnest101",
            block_fn=bottleneck,
            num_blocks=[3, 4, 23, 3],
            deep_stem=True,
            stem_width=64,
            radix=2,
            cardinality=1,
            bottleneck_width=64,
            normalization=normalization,
            last_gamma=last_gamma,
            activation=activation,
            output_indices=output_indices,
            strides=strides,
            dilation_rates=dilation_rates,
            frozen_stages=frozen_stages,
            input_shape=input_shape,
            input_tensor=input_tensor,
            dropblock=dropblock,
            avg_down=True,
            avd=True,
            avd_first=False,
            num_classes=num_classes,
            drop_rate=drop_rate).build_model()


@MODELS.register("ResNeSt152")
def ResNeSt152(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               output_indices=(-1, ), 
               strides=(2, 2, 2, 2, 2), 
               dilation_rates=(1, 1, 1, 1, 1), 
               frozen_stages=(-1, ), 
               input_shape=None, 
               input_tensor=None, 
               dropblock=dict(dropblock_size=7, drop_rate=0.1), 
               last_gamma=False,
               num_classes=1000,
               drop_rate=0.5,
               **kwargs):
        return ResNeSt(
            name="resnest152",
            block_fn=bottleneck,
            num_blocks=[3, 8, 36, 3],
            deep_stem=True,
            stem_width=64,
            radix=2,
            cardinality=1,
            bottleneck_width=64,
            normalization=normalization,
            last_gamma=last_gamma,
            activation=activation,
            output_indices=output_indices,
            strides=strides,
            dilation_rates=dilation_rates,
            frozen_stages=frozen_stages,
            input_shape=input_shape,
            input_tensor=input_tensor,
            dropblock=dropblock,
            avg_down=True,
            avd=True,
            avd_first=False,
            num_classes=num_classes,
            drop_rate=drop_rate).build_model()


@MODELS.register("ResNeSt200")
def ResNeSt200(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               output_indices=(-1, ), 
               strides=(2, 2, 2, 2, 2), 
               dilation_rates=(1, 1, 1, 1, 1), 
               frozen_stages=(-1, ), 
               input_shape=None, 
               input_tensor=None, 
               dropblock=dict(dropblock_size=7, drop_rate=0.1), 
               last_gamma=False,
               num_classes=1000,
               drop_rate=0.5,
               **kwargs):
    return ResNeSt(
        name="resnest200",
        block_fn=bottleneck,
        num_blocks=[3, 24, 36, 3],
        deep_stem=True,
        stem_width=64,
        radix=2,
        cardinality=1,
        bottleneck_width=64,
        normalization=normalization,
        last_gamma=last_gamma,
        activation=activation,
        output_indices=output_indices,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        input_shape=input_shape,
        input_tensor=input_tensor,
        dropblock=dropblock,
        avg_down=True,
        avd=True,
        avd_first=False,
        num_classes=num_classes,
        drop_rate=drop_rate).build_model()


@MODELS.register("ResNeSt269")
def ResNeSt269(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
               activation=dict(activation="relu"),
               output_indices=(-1, ), 
               strides=(2, 2, 2, 2, 2), 
               dilation_rates=(1, 1, 1, 1, 1), 
               frozen_stages=(-1, ), 
               input_shape=None, 
               input_tensor=None, 
               dropblock=dict(dropblock_size=7, drop_rate=0.1), 
               last_gamma=False,
               num_classes=1000,
               drop_rate=0.5,
               **kwargs):
    return ResNeSt(
        name="resnest269",
        block_fn=bottleneck,
        num_blocks=[3, 30, 48, 8],
        deep_stem=True,
        stem_width=64,
        radix=2,
        cardinality=1,
        bottleneck_width=64,
        normalization=normalization,
        last_gamma=last_gamma,
        activation=activation,
        output_indices=output_indices,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        input_shape=input_shape,
        input_tensor=input_tensor,
        dropblock=dropblock,
        avg_down=True,
        avd=True,
        avd_first=False,
        num_classes=num_classes,
        drop_rate=drop_rate).build_model()


def _torch2h5(model, blocks, torch_weight_path):
    import torch
    import numpy as np
    # if model_name == "resnest50":
    #     from resnest.torch import resnest50 as resnest
    # elif model_name == "resnest101":
    #     from resnest.torch import resnest101 as resnest
    # elif model_name == "resnest200":
    #     from resnest.torch import resnest200 as resnest
    # elif model_name == "resnest269":
    #     from resnest.torch import resnest269 as resnest
    # else:
    #     raise ValueError("Wrong name[%s]" % model_name)

    # net = resnest(pretrained=True)

    weight_map = {
        "conv1/1/conv2d/kernel:0"              : "conv1.0.weight",
        "conv1/1/batch_norm/gamma:0"           : "conv1.1.weight",
        "conv1/1/batch_norm/beta:0"            : "conv1.1.bias",
        "conv1/1/batch_norm/moving_mean:0"     : "conv1.1.running_mean",
        "conv1/1/batch_norm/moving_variance:0" : "conv1.1.running_var",
        "conv1/2/conv2d/kernel:0"              : "conv1.3.weight",
        "conv1/2/batch_norm/gamma:0"           : "conv1.4.weight",
        "conv1/2/batch_norm/beta:0"            : "conv1.4.bias",
        "conv1/2/batch_norm/moving_mean:0"     : "conv1.4.running_mean",
        "conv1/2/batch_norm/moving_variance:0" : "conv1.4.running_var",
        "conv1/3/conv2d/kernel:0"              : "conv1.6.weight",
        "conv1/3/batch_norm/gamma:0"           : "bn1.weight",
        "conv1/3/batch_norm/beta:0"            : "bn1.bias",
        "conv1/3/batch_norm/moving_mean:0"     : "bn1.running_mean",
        "conv1/3/batch_norm/moving_variance:0" : "bn1.running_var",
    }
    
    for i, b in enumerate(blocks, 1):
        for j in range(b):
            m = {
                "layer%d/%d/conv1/conv2d/kernel:0" % (i, j)                   : "layer%d.%d.conv1.weight" % (i, j),
                "layer%d/%d/conv1/batch_norm/gamma:0" % (i, j)                : "layer%d.%d.bn1.weight" % (i, j),
                "layer%d/%d/conv1/batch_norm/beta:0" % (i, j)                 : "layer%d.%d.bn1.bias" % (i, j),
                "layer%d/%d/conv1/batch_norm/moving_mean:0" % (i, j)          : "layer%d.%d.bn1.running_mean" % (i, j),
                "layer%d/%d/conv1/batch_norm/moving_variance:0" % (i, j)      : "layer%d.%d.bn1.running_var" % (i, j),
                "layer%d/%d/conv2/conv/conv2d/kernel:0" % (i, j)              : "layer%d.%d.conv2.conv.weight" % (i, j),
                "layer%d/%d/conv2/conv/batch_norm/gamma:0" % (i, j)           : "layer%d.%d.conv2.bn0.weight" % (i, j),
                "layer%d/%d/conv2/conv/batch_norm/beta:0" % (i, j)            : "layer%d.%d.conv2.bn0.bias" % (i, j),
                "layer%d/%d/conv2/conv/batch_norm/moving_mean:0" % (i, j)     : "layer%d.%d.conv2.bn0.running_mean" % (i, j),
                "layer%d/%d/conv2/conv/batch_norm/moving_variance:0" % (i, j) : "layer%d.%d.conv2.bn0.running_var" % (i, j),
                "layer%d/%d/conv2/fc1/conv2d/kernel:0" % (i, j)               : "layer%d.%d.conv2.fc1.weight" % (i, j),
                "layer%d/%d/conv2/fc1/conv2d/bias:0" % (i, j)                 : "layer%d.%d.conv2.fc1.bias" % (i, j),
                "layer%d/%d/conv2/fc1/batch_norm/gamma:0" % (i, j)            : "layer%d.%d.conv2.bn1.weight" % (i, j),
                "layer%d/%d/conv2/fc1/batch_norm/beta:0" % (i, j)             : "layer%d.%d.conv2.bn1.bias" % (i, j),
                "layer%d/%d/conv2/fc1/batch_norm/moving_mean:0" % (i, j)      : "layer%d.%d.conv2.bn1.running_mean" % (i, j),
                "layer%d/%d/conv2/fc1/batch_norm/moving_variance:0" % (i, j)  : "layer%d.%d.conv2.bn1.running_var" % (i, j),
                "layer%d/%d/conv2/fc2/conv2d/kernel:0" % (i, j)               : "layer%d.%d.conv2.fc2.weight" % (i, j),
                "layer%d/%d/conv2/fc2/conv2d/bias:0" % (i, j)                 : "layer%d.%d.conv2.fc2.bias" % (i, j),
                "layer%d/%d/conv3/conv2d/kernel:0" % (i, j)                   : "layer%d.%d.conv3.weight" % (i, j),
                "layer%d/%d/conv3/batch_norm/gamma:0" % (i, j)                : "layer%d.%d.bn3.weight" % (i, j),
                "layer%d/%d/conv3/batch_norm/beta:0" % (i, j)                 : "layer%d.%d.bn3.bias" % (i, j),
                "layer%d/%d/conv3/batch_norm/moving_mean:0" % (i, j)          : "layer%d.%d.bn3.running_mean" % (i, j),
                "layer%d/%d/conv3/batch_norm/moving_variance:0" % (i, j)      : "layer%d.%d.bn3.running_var" % (i, j),
                "layer%d/%d/downsample/conv2d/kernel:0" % (i, j)              : "layer%d.%d.downsample.1.weight" % (i, j),
                "layer%d/%d/downsample/batch_norm/gamma:0" % (i, j)           : "layer%d.%d.downsample.2.weight" % (i, j),
                "layer%d/%d/downsample/batch_norm/beta:0" % (i, j)            : "layer%d.%d.downsample.2.bias" % (i, j),
                "layer%d/%d/downsample/batch_norm/moving_mean:0" % (i, j)     : "layer%d.%d.downsample.2.running_mean" % (i, j),
                "layer%d/%d/downsample/batch_norm/moving_variance:0" % (i, j) : "layer%d.%d.downsample.2.running_var" % (i, j)
            }
            weight_map.update(m)
        
        weight_map["fc/kernel:0"] = "fc.weight"
        weight_map["fc/bias:0"] = "fc.bias"

    net = torch.load(torch_weight_path, map_location=torch.device('cpu'))
    # for k, v in net.items():
    #     if "tracked" in k:
    #         continue
    #     print(k, v.shape) 

    for weight in model.weights:
        name = weight.name
        name = weight_map[name]
        # print(name, weight.shape)

        tw = net[name].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        weight.assign(tw)

    del net



if __name__ == "__main__":
    import cv2

    name = "resnest269"
    model = ResNeSt269(input_shape=(224, 224, 3), output_indices=(-1, ))

    model.load_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % name)
    # _torch2h5(model, [3, 30, 48, 8], "/home/bail/Downloads/%s.pth" % name)

    # with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        # images = tf.image.decode_jpeg(gf.read())

    # images = tf.image.convert_image_dtype(images, tf.float32)
    # images = tf.image.resize(images, (224, 224))[None]
    img = cv2.imread("/home/bail/Documents/plane.jpg")
    img = cv2.resize(img, (224, 224))
    img = img[:, :, ::-1]
    images = tf.convert_to_tensor(img, tf.float32)
    images = tf.expand_dims(images, 0)

    logits = model(images, training=False)
    probs = tf.nn.softmax(logits)
    print(tf.nn.top_k(tf.squeeze(probs), k=5))

    # model.save_weights("/home/bail/Data/data2/pretrained_weights/%s.h5" % name)
    # model.save_weights("/home/bail/Data/data2/pretrained_weights/%s/model.ckpt" % name)
