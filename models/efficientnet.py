import os
import math
import numpy as np
import tensorflow as tf
from collections import namedtuple
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from .common import DepthwiseConvNormActBlock


PARAMS = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        "efficientnet-b8": (2.2, 3.6, 672, 0.5),
        "efficientnet-l2": (4.3, 5.3, 800, 0.5)
    }


GlobalParams = namedtuple("GlobalParams", [
    "batch_norm_momentum", "batch_norm_epsilon", "width_coefficient",
    "depth_coefficient", "depth_divisor", "min_depth", "drop_connect_rate",
    "data_format", "dropout_rate", "num_classes"
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple("BlockArgs", [
    "repeats", "in_filters", "out_filters", "kernel_size",
    "strides", "expand_ratio", "se_ratio", "id_skip", "super_pixel", "trainable"
])
BlockArgs.__new__.__defaults__ = (None, ) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    # orig_filters =filters

    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth

    if multiplier is None:
        return filters

    min_depth = min_depth or divisor
    filters *= multiplier
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)

    if new_filters < 0.9 * filters:
        new_filters += divisor

    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient

    if multiplier is None:
        return repeats

    return int(math.ceil(repeats * multiplier))


class DropConnect(tf.keras.layers.Layer):
    def __init__(self, drop_rate=None, **kwargs):
        super(DropConnect, self).__init__(**kwargs)
        self.drop_rate = drop_rate if drop_rate is not None else 0.

    def build(self, input_shape):
        self.built = True
        super(DropConnect, self).build(input_shape)

    def _drop(self, inputs, drop_rate):
        random_tensor = tf.convert_to_tensor(drop_rate, dtype=inputs.dtype)
        batch_size = tf.shape(inputs)[0]
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.math.floor(random_tensor)

        return tf.divide(inputs, random_tensor) * binary_tensor

    def call(self, inputs, training=None):
        if training or self.drop_rate > 0.:
            return self._drop(inputs, self.drop_rate)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {
            "drop_rate" : self.drop_rate
        }

        base_config = super(DropConnect, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


def conv2d_kernel_initializer(shape, dtype=tf.float32):
    kernel_height, kernel_width, _, out_filters = shape
    fan_out = int(kernel_height * kernel_width * out_filters)

    return tf.random.normal(shape, 0.0, math.sqrt(2. / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=tf.float32):
    init_range = 1.0 / math.sqrt(shape[1])

    return tf.random.uniform(shape, -init_range, init_range, dtype)


def get_weight_names_map(num_blocks, num_no_expand_blocks=1):
    block_map = {
        "head/conv2d/conv2d/kernel": "head/conv2d/kernel",
        "head/conv2d/batch_norm/beta": "head/tpu_batch_normalization/beta",
        "head/conv2d/batch_norm/gamma": "head/tpu_batch_normalization/gamma",
        "head/conv2d/batch_norm/moving_mean": "head/tpu_batch_normalization/moving_mean",
        "head/conv2d/batch_norm/moving_variance": "head/tpu_batch_normalization/moving_variance",
        "head/dense/bias": "head/dense/bias",
        "head/dense/kernel": "head/dense/kernel",
        "stem/conv2d/kernel": "stem/conv2d/kernel",
        "stem/batch_norm/beta": "stem/tpu_batch_normalization/beta",
        "stem/batch_norm/gamma": "stem/tpu_batch_normalization/gamma",
        "stem/batch_norm/moving_mean": "stem/tpu_batch_normalization/moving_mean",
        "stem/batch_norm/moving_variance": "stem/tpu_batch_normalization/moving_variance",
    }

    for i in range(0, num_no_expand_blocks):
        m = {
            "blocks_%d/depthwise/conv2d/depthwise_kernel" % i: "blocks_%d/depthwise_conv2d/depthwise_kernel" % i,
            "blocks_%d/depthwise/batch_norm/beta" % i: "blocks_%d/tpu_batch_normalization/beta" % i,
            "blocks_%d/depthwise/batch_norm/gamma" % i: "blocks_%d/tpu_batch_normalization/gamma" % i,
            "blocks_%d/depthwise/batch_norm/moving_mean" % i: "blocks_%d/tpu_batch_normalization/moving_mean" % i,
            "blocks_%d/depthwise/batch_norm/moving_variance" % i: "blocks_%d/tpu_batch_normalization/moving_variance" % i,
            "blocks_%d/se/conv2d/bias" % i: "blocks_%d/se/conv2d/bias" % i,
            "blocks_%d/se/conv2d/kernel" % i: "blocks_%d/se/conv2d/kernel" % i,
            "blocks_%d/se/conv2d_1/bias" % i: "blocks_%d/se/conv2d_1/bias" % i,
            "blocks_%d/se/conv2d_1/kernel" % i: "blocks_%d/se/conv2d_1/kernel" % i,
            "blocks_%d/project/conv2d/kernel" % i: "blocks_%d/conv2d/kernel" % i,
            "blocks_%d/project/batch_norm/beta" % i: "blocks_%d/tpu_batch_normalization_1/beta" % i,
            "blocks_%d/project/batch_norm/gamma" % i: "blocks_%d/tpu_batch_normalization_1/gamma" % i,
            "blocks_%d/project/batch_norm/moving_mean" % i: "blocks_%d/tpu_batch_normalization_1/moving_mean" % i,
            "blocks_%d/project/batch_norm/moving_variance" % i: "blocks_%d/tpu_batch_normalization_1/moving_variance" % i,
        }
        block_map.update(m)

    for i in range(num_no_expand_blocks, num_blocks):
        m = {
            "blocks_%d/expand/conv2d/kernel" % i: "blocks_%d/conv2d/kernel" % i,
            "blocks_%d/expand/batch_norm/beta" % i: "blocks_%d/tpu_batch_normalization/beta" % i,
            "blocks_%d/expand/batch_norm/gamma" % i: "blocks_%d/tpu_batch_normalization/gamma" % i,
            "blocks_%d/expand/batch_norm/moving_mean" % i: "blocks_%d/tpu_batch_normalization/moving_mean" % i,
            "blocks_%d/expand/batch_norm/moving_variance" % i: "blocks_%d/tpu_batch_normalization/moving_variance" % i,
            "blocks_%d/depthwise/conv2d/depthwise_kernel" % i: "blocks_%d/depthwise_conv2d/depthwise_kernel" % i,
            "blocks_%d/depthwise/batch_norm/beta" % i: "blocks_%d/tpu_batch_normalization_1/beta" % i,
            "blocks_%d/depthwise/batch_norm/gamma" % i: "blocks_%d/tpu_batch_normalization_1/gamma" % i,
            "blocks_%d/depthwise/batch_norm/moving_mean" % i: "blocks_%d/tpu_batch_normalization_1/moving_mean" % i,
            "blocks_%d/depthwise/batch_norm/moving_variance" % i: "blocks_%d/tpu_batch_normalization_1/moving_variance" % i,
            "blocks_%d/se/conv2d/bias" % i: "blocks_%d/se/conv2d/bias" % i,
            "blocks_%d/se/conv2d/kernel" % i: "blocks_%d/se/conv2d/kernel" % i,
            "blocks_%d/se/conv2d_1/bias" % i: "blocks_%d/se/conv2d_1/bias" % i,
            "blocks_%d/se/conv2d_1/kernel" % i: "blocks_%d/se/conv2d_1/kernel" % i,
            "blocks_%d/project/conv2d/kernel" % i: "blocks_%d/conv2d_1/kernel" % i,
            "blocks_%d/project/batch_norm/beta" % i: "blocks_%d/tpu_batch_normalization_2/beta" % i,
            "blocks_%d/project/batch_norm/gamma" % i: "blocks_%d/tpu_batch_normalization_2/gamma" % i,
            "blocks_%d/project/batch_norm/moving_mean" % i: "blocks_%d/tpu_batch_normalization_2/moving_mean" % i,
            "blocks_%d/project/batch_norm/moving_variance" % i: "blocks_%d/tpu_batch_normalization_2/moving_variance" % i,
        }

        block_map.update(m)

    return block_map
    

class MBConvBlock(tf.keras.Model):
    def __init__(self,
                 global_params,
                 block_args,
                 normalization,
                 activation=dict(activation="swish"),
                 drop_connect_rate=None,
                 trainable=True,
                 name=None):
        super(MBConvBlock, self).__init__(name=name)
        self.expand_ratio = block_args.expand_ratio
        data_format = global_params.data_format

        _axis = normalization["axis"]
        self.mean_axis = [1, 2] if _axis == -1 or _axis == 3 else [2, 3]
        filters = block_args.in_filters * self.expand_ratio
        if self.expand_ratio != 1:
            self.expand = ConvNormActBlock(filters=filters, 
                                      kernel_size=(1, 1), 
                                      trainable=trainable, 
                                      normalization=normalization,
                                      kernel_initializer=conv2d_kernel_initializer,
                                      activation=activation,
                                      name="expand")

        self.depthwise = DepthwiseConvNormActBlock(kernel_size=block_args.kernel_size,
                                              strides=block_args.strides,
                                              data_format=data_format,
                                              normalization=normalization,
                                              activation=activation,
                                              kernel_initializer=conv2d_kernel_initializer,
                                              name="depthwise")
       
        has_se = block_args.se_ratio is not None and 0 < block_args.se_ratio < 1
        if has_se:
            squeezed_filters = max(1, int(block_args.in_filters * block_args.se_ratio))
            self.se_reduce = tf.keras.layers.Conv2D(filters=squeezed_filters,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding="same",
                                                    data_format=data_format,
                                                    use_bias=True,
                                                    kernel_initializer=conv2d_kernel_initializer,
                                                    trainable=trainable,
                                                    name="se/conv2d")
            self.swish = tf.keras.layers.Activation("swish")
            self.se_expand = tf.keras.layers.Conv2D(filters=filters,
                                                    kernel_size=(1, 1),
                                                    strides=(1, 1),
                                                    padding="same",
                                                    data_format=data_format,
                                                    use_bias=True,
                                                    trainable=trainable,
                                                    kernel_initializer=conv2d_kernel_initializer,
                                                    name="se/conv2d_1")
        
        self.project = ConvNormActBlock(block_args.out_filters,
                                   kernel_size=(1, 1),
                                   trainable=trainable,
                                   normalization=normalization,
                                   activation=None,
                                   kernel_initializer=conv2d_kernel_initializer,
                                   name="project")
        
        self._id_skip = False
        if block_args.id_skip:
            if all(s == 1 for s in block_args.strides) and block_args.in_filters == block_args.out_filters:
                if drop_connect_rate > 0:
                    self.drop_connect = DropConnect(drop_connect_rate, name=name + "/drop_connect")
                self._id_skip = True
        self._has_se = has_se
    
    def call(self, inputs, training=None):
        x = inputs
        
        if hasattr(self, "expand"):
            x = self.expand(x, training=training)
        
        x = self.depthwise(x, training=training)
        
        if self._has_se:
            x_squeezed = tf.reduce_mean(x, self.mean_axis, keepdims=True)
            x_squeezed = self.se_reduce(x_squeezed)
            x_squeezed = self.swish(x_squeezed)
            x_squeezed = self.se_expand(x_squeezed)
            x = tf.nn.sigmoid(x_squeezed) * x
        
        x = self.project(x, training=training)

        if hasattr(self, "drop_connect"):
            x = self.drop_connect(x, training=training)
        
        if self._id_skip:
            x += inputs
        
        return x


class EfficientNet(Model):
    def _get_global_params(self, name, data_format, num_classes):
        return GlobalParams(
            batch_norm_momentum=0.9,
            batch_norm_epsilon=1e-3,
            width_coefficient=PARAMS[name][0],
            depth_coefficient=PARAMS[name][1],
            depth_divisor=8,
            min_depth=None,
            drop_connect_rate=0.2,
            data_format=data_format,
            dropout_rate=PARAMS[name][-1],
            num_classes=num_classes)

    def _get_block_args(self):
        return [
            BlockArgs(1, 32, 16, (3, 3), (1, 1), 1, 0.25, True),
            BlockArgs(2, 16, 24, (3, 3), (2, 2), 6, 0.25, True),
            BlockArgs(2, 24, 40, (5, 5), (2, 2), 6, 0.25, True),
            BlockArgs(3, 40, 80, (3, 3), (2, 2), 6, 0.25, True),
            BlockArgs(3, 80, 112, (5, 5), (1, 1), 6, 0.25, True),
            BlockArgs(4, 112, 192, (5, 5), (2, 2), 6, 0.25, True),
            BlockArgs(1, 192, 320, (3, 3), (1, 1), 6, 0.25, True)
        ]

    def __init__(self,
                 name,
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                 activation=dict(activation="swish"), 
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 dropblock=None, 
                 input_shape=None,
                 input_tensor=None,
                 **kwargs):
        data_format = tf.keras.backend.image_data_format()
        self.batch_normalization_axis = 3 if data_format == 'channels_last' else 1
        default_size = PARAMS[name][2]
        dropout_connect_rate = PARAMS[name][3]
        default_shape = [default_size, default_size, 3] if self.batch_normalization_axis == 3 else [3, default_size, default_size]
        input_shape = input_shape or default_shape

        super(EfficientNet, self).__init__(name=name,
                                           convolution=convolution,
                                           normalization=normalization, 
                                           activation=activation, 
                                           output_indices=output_indices, 
                                           strides=strides, 
                                           dilation_rates=dilation_rates, 
                                           frozen_stages=frozen_stages, 
                                           dropblock=dropblock, 
                                           input_shape=input_shape,
                                           input_tensor=input_tensor,
                                           **kwargs)
        self.backbone_name = name
        self.data_format = data_format

        self.global_params = self._get_global_params(name, self.data_format, self.num_classes)
        self.block_args = self._get_block_args()

        self._drop_connect_rate = dropout_connect_rate

        self.num_blocks = 0
        for args in self.block_args:
            self.num_blocks += round_repeats(args.repeats, global_params=self.global_params)
        
        self._num_no_expand_blocks = 0
    
    @property
    def blocks(self):
        blocks = []
        for i, args in enumerate(self.block_args):
            assert args.repeats >= 1
            # assert args.super_pixel in [0, 1, 2]
            in_filters = round_filters(args.in_filters, self.global_params)
            out_filters = round_filters(args.out_filters, self.global_params)
            
            args = args._replace(in_filters=in_filters,
                                 out_filters=out_filters,
                                 repeats=round_repeats(args.repeats, self.global_params),
                                 trainable=i + 2 not in self.frozen_stages)
            blocks.append(args)
            if args.repeats > 1:
                args = args._replace(in_filters=out_filters, strides=(1, 1))
            for i in range(args.repeats - 1):
                blocks.append(args)
        
        return blocks
    
    def build_model(self):
        def _norm(inp):
            inp -= (tf.convert_to_tensor(self._rgb_mean * 255., inp.dtype))
            inp /= (tf.convert_to_tensor(self._rgb_std * 255., inp.dtype))
            
            return inp

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)
        x = ConvNormActBlock(filters=round_filters(32, self.global_params),
                        kernel_size=(3, 3),
                        strides=(2, 2),
                        data_format=self.data_format,
                        normalization=self.normalization,
                        kernel_initializer=conv2d_kernel_initializer,
                        trainable=1 not in self.frozen_stages,
                        activation=self.activation,
                        name=self.name + "/stem")(x)

        block_outputs = []
        for idx, b_args in enumerate(self.blocks):
            drop_rate = self._drop_connect_rate
            is_reduction = False
                
            if b_args.super_pixel == 1 and idx == 0:
                block_outputs.append(x)
            elif (idx == self.num_blocks - 1) or self.blocks[idx+1].strides[0] > 1:
                is_reduction = True
            if drop_rate:
                drop_rate = 1.0 - drop_rate * float(idx) / self.num_blocks
            
            if b_args.expand_ratio == 1:
                self._num_no_expand_blocks += 1

            x = MBConvBlock(global_params=self.global_params,
                            block_args=b_args,
                            normalization=self.normalization,
                            drop_connect_rate=drop_rate,
                            trainable=b_args.trainable,
                            name=self.name + "/blocks_%d" % idx)(x)
            
            if is_reduction:
                block_outputs.append(x)
                
        if -1 in self.output_indices:
            # Head part.
            x = ConvNormActBlock(filters=round_filters(1280, self.global_params),
                            kernel_size=[1, 1],
                            strides=[1, 1],
                            data_format=self.data_format,
                            normalization=self.normalization,
                            kernel_initializer=conv2d_kernel_initializer,
                            name=self.name + "/head/conv2d")(x)
           
            x = tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, 
                                                       name=self.name + "/head/global_avg_pooling")(x)
            x = tf.keras.layers.Dropout(self.global_params.dropout_rate, name=self.name + "/head/dropout")(x)
            x = tf.keras.layers.Dense(self.global_params.num_classes,
                                      kernel_initializer=dense_kernel_initializer, 
                                      name=self.name + "/head/dense")(x)
            outputs = x
            
        else:
            outputs = [block_outputs[i - 1] for i in self.output_indices]
            
        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def init_weights(self, model, pretrained_weight_path=None):
        if pretrained_weight_path is not None:
            pretrained_weights = tf.train.latest_checkpoint(pretrained_weight_path)
            assert pretrained_weights is not None, "Error! Please check path {}".format(pretrained_weight_path)
            use_exponential_moving_average = False
            for w in tf.train.list_variables(pretrained_weights):
                if "ExponentialMovingAverage" in w[0]:
                    use_exponential_moving_average = True
                    # print(w[0], w[1])

            names_map = get_weight_names_map(self.num_blocks, self._num_no_expand_blocks)
            for weight in model.weights:
                name = weight.name.split(":")[0]
                name = name.replace(self.name + "/", "")

                pname = self.name + "/" + names_map[name]
                
                if use_exponential_moving_average:
                    pname += "/ExponentialMovingAverage"
                
                try:
                    pretrained_weight = tf.train.load_variable(pretrained_weights, pname)
                    weight.assign(pretrained_weight)
                except Exception as e:
                    print(str(e))


@MODELS.register("EfficientNetB0")
def EfficientNetB0(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=(-1, ), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b0",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b0"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse
    return net


@MODELS.register("EfficientNetB1")
def EfficientNetB1(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b1",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b1"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


@MODELS.register("EfficientNetB2")
def EfficientNetB2(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b2",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b2"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


@MODELS.register("EfficientNetB3")
def EfficientNetB3(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b3",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b3"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net

@MODELS.register("EfficientNetB4")
def EfficientNetB4(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b4",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b4"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


@MODELS.register("EfficientNetB5")
def EfficientNetB5(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b5",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b5"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


@MODELS.register("EfficientNetB6")
def EfficientNetB6(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b6",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b6"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


@MODELS.register("EfficientNetB7")
def EfficientNetB7(convolution='conv2d', 
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
                   activation=dict(activation="swish"), 
                   output_indices=(3, 4), 
                   strides=(2, 2, 2, 2, 2), 
                   dilation_rates=(1, 1, 1, 1, 1), 
                   frozen_stages=( -1), 
                   dropblock=None, 
                   input_tensor=None,
                   **kwargs):
    net = EfficientNet(name="efficientnet-b7",
                       convolution=convolution, 
                       normalization=normalization,
                       activation=activation, 
                       output_indices=output_indices, 
                       strides=strides, 
                       dilation_rates=dilation_rates, 
                       frozen_stages=frozen_stages, 
                       dropblock=dropblock, 
                       input_shape=PARAMS["efficientnet-b7"][2],
                       input_tensor=input_tensor,
                       **kwargs).build_model()
    # net.fuse = fuse 
    return net


if __name__ == "__main__":
    import cv2
    from .common import fuse
    
    shape = 224
    name = "efficientnet-b0"
    checkpoint_dir = "/Users/bailang/Downloads/pretrained_weights/noisy_student_%s" % name
    efficientnet = EfficientNet(name=name, output_indices=[-1, ])
    model = efficientnet.build_model()
    
    efficientnet.init_weights(model, checkpoint_dir)

    # fuse(model, MBConvBlock)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        image = tf.image.decode_jpeg(gf.read())

    image = tf.image.resize(image, (shape, shape))
    images = tf.cast(image, tf.uint8)[None]
    probs = tf.nn.softmax(model(images, training=False))
    # model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    
    print(tf.nn.top_k(tf.squeeze(probs), k=5))
