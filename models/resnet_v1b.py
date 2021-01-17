import os
import tensorflow as tf
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from core.layers import build_activation


class BottleNeckV1B(tf.keras.Model):
    def __init__(self,
                 in_filters,
                 out_filters,
                 strides=1,
                 dilation_rate=1,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 trainable=True,
                 dropblock=None,
                 avg_down=False,
                 last_gamma_init_zero=False,
                 data_format=None):
        """A residual block.

            Args:
                filters: integer, filters of the bottleneck layer.
                strides: default 1, stride of the first layer.
                dilation_rate: default 1, dilation rate in 3x3 convolution.
                activation: the activation layer name.
                trainable: does this block is trainable.
                normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
                dropblock: the arguments in DropBlock2D
                use_conv_shortcut: default True, use convolution shortcut if True,
                    otherwise identity shortcut.
            Returns:
                Output tensor for the residual block.
        """
        self.conv1 = ConvNormActBlock()
        x = build_convolution(convolution,
                            filters=out_filters // 4,
                            kernel_size=3,
                            strides=strides,
                            padding="SAME",
                            use_bias=False,
                            dilation_rate=dilation_rate,
                            data_format=data_format,
                            trainable=trainable,
                            name="layers%d/conv%d" % (stage_index, block_index * 3 + 1))(x)
        x = build_normalization(**normalization, 
                                gamma_initializer=tf.keras.initializers.Zeros() if last_gamma_init_zero else tf.keras.initializers.Ones(),
                                name="layers%d/%s%d" % (stage_index, norm, block_index * 3 + 1))(x)
        x = tf.keras.layers.Activation(**activation, 
                                    name="layers%d/%s%d" % (stage_index, act, block_index * 3 + 1))(x)
        if dropblock is not None:
            x = DropBlock2D(**dropblock, 
                            data_format=data_format, 
                            name="layers%d/dropblock2d%d" % (stage_index, block_index * 3 + 1))(x)

        x = build_convolution(convolution,
                            filters=out_filters,
                            kernel_size=1,
                            trainable=trainable,
                            data_format=data_format,
                            use_bias=False,
                            name="layers%d/conv%d" % (stage_index, block_index * 3 + 2))(x)
        x = build_normalization(**normalization, 
                                name="layers%d/%s%d" % (stage_index, norm, block_index * 3 + 2))(x)
        if dropblock is not None:
            x = DropBlock2D(**dropblock, 
                            data_format=data_format, 
                            name="layers%d/dropblock2d%d" % (stage_index, block_index * 3 + 2))(x)
        
        shortcut = inputs
        if strides != 1 or in_filters != out_filters:
            if avg_down:
                if dilation_rate == 1:
                    shortcut = tf.keras.layers.AvgPool2D(pool_size=strides, 
                                                        strides=strides, 
                                                        padding="same", 
                                                        data_format=data_format, 
                                                        name="down%d/avg_pool" % stage_index)(shortcut)
                else:
                    shortcut = tf.keras.layers.AvgPool2D(pool_size=1, 
                                                        strides = 1, 
                                                        padding="same", 
                                                        data_format=data_format,
                                                        name="down%d/avg_pool" % stage_index)(shortcut)
                shortcut = tf.keras.layers.Conv2D(kernel_size=1, 
                                                filters=out_filters,
                                                strides=1,
                                                padding="same",
                                                use_bias=False,
                                                trainable=trainable,
                                                data_format=data_format,
                                                name="down%d/conv0" % stage_index)(shortcut)
                shortcut = build_normalization(**normalization,
                                            name="down%d/%s0" % (stage_index, norm))(shortcut)
            else:
                shortcut = tf.keras.layers.Conv2D(filters=out_filters, 
                                                kernel_size=1, 
                                                strides=strides,
                                                use_bias=False,
                                                padding="same",
                                                trainable=trainable,
                                                data_format=data_format,
                                                name="down%d/conv0" % stage_index)(shortcut)
                shortcut = build_normalization(**normalization, 
                                                name="down%d/%s0" % (stage_index, norm))(shortcut)
            
            if dropblock is not None:
                shortcut = DropBlock2D(**dropblock,
                                    data_format=data_format,
                                    name="down%s/dropblock2d0" % stage_index)(shortcut)

        x = tf.keras.layers.Add(name="layers%d/sum%d" % (stage_index, block_index))([shortcut, x])
        x = tf.keras.layers.Activation(**activation, name="layers%d/%s%d" % (stage_index, act, block_index * 3 + 2))(x)



class ResNetV1B(Backbone):
    def __init__(self, 
                 name, 
                 deep_stem=False,
                 block_fn=bottleneck_v1b,
                 num_blocks=(3, 4, 6, 3),
                 stem_filters=32,
                 convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 avg_down=False,
                 num_classes=1000,
                 drop_rate=0.5,
                 **kwargs):
        super(ResNetV1B, self).__init__(name,
                                        convolution=convolution, 
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
                                        drop_rate=drop_rate,
                                        **kwargs)
        self.deep_stem = deep_stem
        self.block_fn = block_fn
        self.num_blocks = num_blocks
        self.stem_filters = stem_filters
        self.last_gamma = last_gamma
        self.avg_down = avg_down

    def build_model(self):
        norm = self.normalization.get("normalization")
        act = self.activation.get("activation")

        def _norm(inp):
            inp -= tf.constant([0.485, 0.456, 0.406], tf.float32, [1, 1, 1, 3])
            inp /= tf.constant([0.229, 0.224, 0.225], tf.float32, [1, 1, 1, 3])

            return inp

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)

        if not self.deep_stem:
            x = tf.keras.layers.Conv2D(filters=64, 
                                       kernel_size=(7, 7), 
                                       strides=(2, 2), 
                                       padding="same", 
                                       data_format=self.data_format, 
                                       use_bias=False,
                                       name="conv0")(x)
        else:
            x = tf.keras.layers.Conv2D(filters=self.stem_filters, 
                                       kernel_size=(3, 3), 
                                       strides=(2, 2), 
                                       padding="same", 
                                       data_format=self.data_format, 
                                       use_bias=False,
                                       name="conv0")(x)
            x = build_normalization(**self.normalization, name="%s0" % norm)(x)
            x = build_activation(**self.activation, name="%s0" % act)(x)
            x = tf.keras.layers.Conv2D(filters=self.stem_filters, 
                                       kernel_size=(3, 3), 
                                       strides=(1, 1), 
                                       padding="same", 
                                       data_format=self.data_format, 
                                       use_bias=False,
                                       name="conv1")(x)
            x = build_normalization(**self.normalization, name="%s1" % norm)(x)
            x = build_activation(**self.activation, name="%s1" % act)(x)
            x = tf.keras.layers.Conv2D(filters=self.stem_filters * 2, 
                                       kernel_size=(3, 3), 
                                       strides=(1, 1), 
                                       padding="same", 
                                       data_format=self.data_format, 
                                       use_bias=False,
                                       name="conv2")(x)
        x = build_normalization(**self.normalization, name="%s2" % norm if self.deep_stem else "%s0" % norm)(x)
        x = build_activation(**self.activation, name="%s2" % act if self.deep_stem else "%s0" % act)(x)
        outputs = [x]
        x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), 
                                      strides=self.strides[1], 
                                      padding="same", 
                                      data_format=self.data_format,
                                      name="max_pool0")(x)
        x = self._make_layers(x, 1, self.block_fn, 64, 256, self.num_blocks[0], 1, self.dilation_rates[1], 2 not in self.frozen_stages)
        outputs.append(x)
        x = self._make_layers(x, 2, self.block_fn, 256, 512, self.num_blocks[1], self.strides[2], self.dilation_rates[2], 3 not in self.frozen_stages)
        outputs.append(x)
        x = self._make_layers(x, 3, self.block_fn, 512, 1024, self.num_blocks[2], self.strides[3], self.dilation_rates[3], 4 not in self.frozen_stages)
        outputs.append(x)
        x = self._make_layers(x, 4, self.block_fn, 1024, 2048, self.num_blocks[3], self.strides[4], self.dilation_rates[4], 5 not in self.frozen_stages)
        outputs.append(x)
        
        if -1 not in self.output_indices:
            outputs = [outputs[i-1] for i in self.output_indices]
        else:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
            if self.drop_rate and self.drop_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.drop_rate, name="dropout0")(x)
            
            outputs = tf.keras.layers.Dense(units=self.num_classes, name="dense0")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)

    def _make_layers(self, x, stage_index, block_fn, in_filters, out_filters, num_blocks, strides=1, dilation_rate=1, trainable=True):
        x = block_fn(x,
                     stage_index=stage_index,
                     block_index=0,
                     convolution= self.convolution,
                     in_filters=in_filters,
                     out_filters=out_filters,
                     strides=strides,
                     dilation_rate=dilation_rate,
                     normalization=self.normalization,
                     activation=self.activation,
                     trainable=trainable,
                     dropblock=self.dropblock,
                     last_gamma_init_zero=self.last_gamma,
                     avg_down=self.avg_down,
                     data_format=self.data_format)
        for i in range(1, num_blocks):
            x = block_fn(x,
                         stage_index=stage_index,
                         block_index=i,
                         convolution= self.convolution,
                         in_filters=out_filters,
                         out_filters=out_filters,
                         strides=1,
                         dilation_rate=dilation_rate,
                         normalization=self.normalization,
                         activation=self.activation,
                         trainable=trainable,
                         dropblock=self.dropblock,
                         last_gamma_init_zero=self.last_gamma,
                         avg_down=self.avg_down,
                         data_format=self.data_format)

        return x


@BACKBONES.register("ResNet50V1C")
def ResNet50V1C(convolution='conv2d', 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(3, 4), 
                strides=(2, 2, 2, 2, 2), 
                dilation_rates=(1, 1, 1, 1, 1), 
                frozen_stages=(-1, ), 
                input_shape=None, 
                input_tensor=None, 
                dropblock=None, 
                last_gamma=False,
                num_classes=1000,
                drop_rate=0.5):
    return ResNetV1B("resnet50_v1c",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet101V1C")
def ResNet101V1C(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet101_v1c",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet152V1C")
def ResNet152V1C(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet152_v1c",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     deep_stem=True,
                     last_gamma=last_gamma,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=32,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet50V1D")
def ResNet50V1D(convolution='conv2d', 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(3, 4), 
                strides=(2, 2, 2, 2, 2), 
                dilation_rates=(1, 1, 1, 1, 1), 
                frozen_stages=(-1, ), 
                input_shape=None, 
                input_tensor=None, 
                dropblock=None, 
                last_gamma=False,
                num_classes=1000,
                drop_rate=0.5):
    return ResNetV1B("resnet50_v1d",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     deep_stem=True,
                     last_gamma=last_gamma,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=32,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet101V1D")
def ResNet101V1D(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet101_v1d",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=32,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model( )
 

@BACKBONES.register("ResNet152V1D")
def ResNet152V1D(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet152_v1d",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=32,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet50V1E")
def ResNet50V1E(convolution='conv2d', 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(3, 4), 
                strides=(2, 2, 2, 2, 2), 
                dilation_rates=(1, 1, 1, 1, 1), 
                frozen_stages=(-1, ), 
                input_shape=None, 
                input_tensor=None, 
                dropblock=None, 
                last_gamma=False,
                num_classes=1000,
                drop_rate=0.5):
    return ResNetV1B("resnet50_v1e",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=64,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet101V1E")
def ResNet101V1E(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet101_v1e",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=64,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet152V1E")
def ResNet152V1E(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet152_v1e",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock,
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=64,
                     avg_down=True,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet50V1S")
def ResNet50V1S(convolution='conv2d', 
                normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                activation=dict(activation="relu"),
                output_indices=(3, 4), 
                strides=(2, 2, 2, 2, 2), 
                dilation_rates=(1, 1, 1, 1, 1), 
                frozen_stages=(-1, ), 
                input_shape=None, 
                input_tensor=None, 
                dropblock=None, 
                last_gamma=False,
                num_classes=1000,
                drop_rate=0.5,
                **kwargs):
    return ResNetV1B("resnet50_v1s",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 6, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet101V1S")
def ResNet101V1S(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):
    return ResNetV1B("resnet101_v1s",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 4, 23, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


@BACKBONES.register("ResNet152V1S")
def ResNet152V1S(convolution='conv2d', 
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 dropblock=None, 
                 last_gamma=False,
                 num_classes=1000,
                 drop_rate=0.5):

    return ResNetV1B("resnet152_v1s",
                     convolution=convolution, 
                     normalization=normalization, 
                     activation=activation, 
                     output_indices=output_indices, 
                     strides=strides, 
                     dilation_rates=dilation_rates, 
                     frozen_stages=frozen_stages, 
                     input_shape=input_shape, 
                     input_tensor=input_tensor,
                     dropblock=dropblock, 
                     last_gamma=last_gamma,
                     deep_stem=True,
                     block_fn=bottleneck_v1b,
                     num_blocks=[3, 8, 36, 3],
                     stem_filters=64,
                     num_classes=num_classes,
                     drop_rate=drop_rate).build_model()


def _mxnet2h5(model, name, out_path):
    # from mxnet import nd, image
    from gluoncv.model_zoo import get_model
    # from gluoncv.data.transforms.presets.imagenet import transform_eval


    m_name = name
    m_name = m_name.replace("18", "")
    m_name = m_name.replace("35", "")
    m_name = m_name.replace("50", "")
    m_name = m_name.replace("101", "")
    m_name = m_name.replace("152", "")
    m_name = m_name.replace("_", "")

    # img = image.imread("/home/bail/Documents/pandas.jpg")
    # img = transform_eval(img)
    print(name)
    net = get_model(name, pretrained=True)
    # pred = net(img)
    # topK = 5
    # ind = nd.topk(pred, k=topK)[0].astype('int')
    # print('The input picture is classified to be')
    # for i in range(topK):
    #     print('\t[%s], with probability %.3f.'%
    #           (ind[i].asscalar(), nd.softmax(pred)[0][ind[i]].asscalar()))
    m_weights = net.collect_params()

    for weight in model.weights:
        # print(m_name, tw.name, tw.shape, mw.name, mw.shape)
        
        mw_n = weight.name.split(":")[0]
        mw_n = mw_n.replace("/", "_")
        if "kernel" in mw_n:
            mw_n = mw_n.replace("kernel", "weight")
        if "batch_norm" in mw_n:
            mw_n = mw_n.replace("batch_norm", "batchnorm")
        if "group_norm" in mw_n:
            mw_n = mw_n.replace("group_norm", "groupnorm")
        if "moving_mean" in mw_n:
            mw_n = mw_n.replace("moving_mean", "running_mean")
        if "moving_variance" in mw_n:
            mw_n = mw_n.replace("moving_variance", "running_var")
        mw_n = m_name + "_" + mw_n

        mw = m_weights[mw_n].data().asnumpy()
        if len(mw.shape) == 4:
            mw = mw.transpose((2, 3, 1, 0))
        
        if len(mw.shape) == 2:
            mw = mw.transpose((1, 0))
        weight.assign(mw)

    del net

    model.save_weights(out_path)


if __name__ == "__main__":
    name = "resnet152v1d"
    model = ResNet152V1E(input_shape=(224, 224, 3), output_indices=(-1, ))
    # _mxnet2h5(model, name, "/home/bail/Workspace/pretrained_weights/resnet152v1s/resnet52v1s.ckpt")
    model.load_weights("/home/bail/Workspace/pretrained_weights/%s/%s.ckpt" % (name, name))

    with tf.io.gfile.GFile("/home/bail/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.convert_image_dtype(images, tf.float32)
    images = tf.image.resize(images, (224, 224))[None]
    logits = model(images, training=False)
    probs = tf.nn.softmax(logits)
    print(tf.nn.top_k(tf.squeeze(probs), k=5))

    model.save_weights("/home/bail/Workspace/pretrained_weights/%s.h5" % name)
