import os
import tensorflow as tf
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from core.layers import build_activation


class BasicBlock(tf.keras.Model):
    """
    Basic Residual block
    
    Args:
        filters(int): integer, filters of the bottleneck layer.
        strides(int): default 1, stride of the first layer.
        dilation_rate(int): default 1, dilation rate in 3x3 convolution.
        data_format(str): default channels_last,
        normalization(dict): the normalization name and hyper-parameters, e.g.
            dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True), 
            dict(normalization="group_norm", epsilon=1e-3, axis=-1) etc.
        activation: the activation layer name.
        trainable: does this block is trainable.
        dropblock: the arguments in DropBlock2D
        use_conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label, default None.
    """
    expansion = 1

    def __init__(self,
                 filters,
                 strides=1,
                 dilation_rate=1,
                 data_format="channels_last",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 trainable=True,
                 dropblock=None,
                 use_conv_shortcut=False,
                 name=None):
        super(BasicBlock, self).__init__(name=name)
        self.conv1 = ConvNormActBlock(filters=filters,
                                      kernel_size=3,
                                      strides=strides,
                                      data_format=data_format,
                                      dilation_rate=1 if strides > 1 else dilation_rate,
                                      trainable=trainable,
                                      normalization=normalization,
                                      activation=activation,
                                      dropblock=dropblock,
                                      name="conv1")
        self.conv2 = ConvNormActBlock(filters=filters,
                                      kernel_size=3,
                                      strides=1,
                                      data_format=data_format,
                                      dilation_rate=dilation_rate,
                                      trainable=trainable,
                                      normalization=normalization,
                                      activation=None,
                                      dropblock=dropblock,
                                      name="conv2")
        self.act = build_activation(**activation, name=activation["activation"])
        if use_conv_shortcut:
            self.shortcut = ConvNormActBlock(filters=filters,
                                             kernel_size=1,
                                             strides=strides,
                                             data_format=data_format,
                                             trainable=trainable,
                                             normalization=normalization,
                                             activation=None,
                                             dropblock=dropblock,
                                             name="shortcut")
    
    def call(self, inputs, training=None):
        shortcut = inputs
        if hasattr(self, "shortcut"):
            shortcut = self.shortcut(shortcut, training=training)
        
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x += shortcut
        x = self.act(x)

        return x


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self,
                 filters,
                 strides=1,
                 dilation_rate=1,
                 data_format="channels_last",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 trainable=True,
                 dropblock=None,
                 use_conv_shortcut=True,
                 name=None):
        """A residual block.

            Args:
                filters: integer, filters of the bottleneck layer.
                convolution: The convolution type.
                strides: default 1, stride of the first layer.
                dilation_rate: default 1, dilation rate in 3x3 convolution.
                data_format: default channels_last,
                activation: the activation layer name.
                trainable: does this block is trainable.
                normalization: the normalization, e.g. "batch_norm", "group_norm" etc.
                dropblock: the arguments in DropBlock2D
                use_conv_shortcut: default True, use convolution shortcut if True,
                    otherwise identity shortcut.
                name: string, block label.
        """
        super(Bottleneck, self).__init__(name=name)
        if use_conv_shortcut is True:
            self.shortcut = ConvNormActBlock(filters=self.expansion * filters,
                                             kernel_size=1,
                                             strides=strides,
                                             data_format=data_format,
                                             trainable=trainable,
                                             dropblock=dropblock,
                                             normalization=normalization,
                                             activation=None,
                                             name="shortcut")
        self.conv1 = ConvNormActBlock(filters=filters,
                                      kernel_size=1,
                                      strides=1,
                                      trainable=trainable,
                                      dropblock=dropblock,
                                      data_format=data_format,
                                      normalization=normalization,
                                      activation=activation,
                                      name="conv1")
        self.conv2 = ConvNormActBlock(filters=filters,
                                kernel_size=3,
                                strides=strides,
                                dilation_rate=dilation_rate if strides == 1 else 1,
                                trainable=trainable,
                                data_format=data_format,
                                normalization=normalization,
                                activation=activation,
                                name="conv2")
        self.act = build_activation(**activation, name=activation["activation"])
        self.conv3 = ConvNormActBlock(filters=self.expansion * filters,
                                      kernel_size=1,
                                      trainable=trainable,
                                      data_format=data_format,
                                      normalization=normalization,
                                      activation=None,
                                      name="conv3")

    def call(self, inputs, training=None):
        if hasattr(self, "shortcut"):
            shortcut = self.shortcut(inputs, training=training)
        else:
            shortcut = inputs
        
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x += shortcut
        x = self.act(x)
        return x


class ResNet(Model):
    def __init__(self,
                 name,
                 blocks,
                 block_fn,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(-1, ),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 **kwargs):
        super(ResNet, self).__init__(name=name,
                                     normalization=normalization,
                                     activation=activation,
                                     output_indices=output_indices,
                                     strides=strides,
                                     dilation_rates=dilation_rates,
                                     frozen_stages=frozen_stages,
                                     dropblock=dropblock,
                                     num_classes=num_classes,
                                     input_shape=input_shape,
                                     input_tensor=input_tensor,
                                     drop_rate=drop_rate,
                                     **kwargs)
        self.blocks = blocks
        self.block_fn = block_fn

    def build_model(self):
        def _norm(inp):
            inp -= (tf.convert_to_tensor(self._rgb_mean * 255., inp.dtype))
            inp /= (tf.convert_to_tensor(self._rgb_std * 255., inp.dtype))
            return inp

        x = tf.keras.layers.Lambda(function=_norm, name="norm_input")(self.img_input)
        x = ConvNormActBlock(filters=64,
                             kernel_size=(7, 7),
                             strides=self.strides[0],
                             dilation_rate=self.dilation_rates[0],
                             trainable=1 not in self.frozen_stages,
                             kernel_initializer="he_normal",
                             normalization=self.normalization,
                             name="conv1")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x1 = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", name="pool1")(x)
        self.in_filters = 64 

        trainable = 2 not in self.frozen_stages
        x2 = self.stack(x1, 64, 1, self.dilation_rates[1], trainable, self.blocks[0], "layer1")
        trainable = 3 not in self.frozen_stages
        x3 = self.stack(x2, 128, self.strides[2], self.dilation_rates[2], trainable, self.blocks[1], "layer2")
        trainable = 4 not in self.frozen_stages
        x4 = self.stack(x3, 256, self.strides[3], self.dilation_rates[3], trainable, self.blocks[2], "layer3")
        trainable = 5 not in self.frozen_stages
        x5 = self.stack(x4, 512, self.strides[4], self.dilation_rates[4], trainable, self.blocks[3], "layer4")

        if -1 in self.output_indices:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format, name="avg_pool")(x5)
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
            outputs = tf.keras.layers.Dense(self.num_classes, name="logits")(x)
        else:
            outputs = [o for i, o in enumerate([x1, x2, x3, x4, x5]) if i + 1 in self.output_indices]

        model = tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)

        return model

    def stack(self, x, filters, strides, dilation_rate, trainable, blocks, name=None):
        use_conv_shortcut = False
        if strides != 1 or self.in_filters != filters * self.block_fn.expansion:
            use_conv_shortcut = True
        x = self.block_fn(filters=filters,
                          strides=strides,
                          dilation_rate=dilation_rate,
                          normalization=self.normalization,
                          activation=self.activation,
                          trainable=trainable,
                          dropblock=self.dropblock,
                          use_conv_shortcut=use_conv_shortcut,
                          name=name + "/0")(x)
        for i in range(1, blocks):
            x = self.block_fn(filters=filters,
                              strides=1,
                              dilation_rate=dilation_rate,
                              normalization=self.normalization,
                              activation=self.activation,
                              trainable=trainable,
                              dropblock=self.dropblock,
                              use_conv_shortcut=False,
                              name=name + "/%d" % i)(x)
        return x


@MODELS.register("ResNet18")
def ResNet18(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
             activation=dict(activation="relu"),
             output_indices=(-1, ),
             strides=(2, 2, 2, 2, 2),
             dilation_rates=(1, 1, 1, 1, 1),
             frozen_stages=(-1, ),
             dropblock=None,
             num_classes=1000,
             drop_rate=0.5,
             input_shape=(224, 224, 3),
             input_tensor=None,
             **kwargs):
    return ResNet(name="resnet18",
                  blocks=[2, 2, 2, 2],
                  block_fn=BasicBlock,
                  normalization=normalization,
                  activation=activation,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  drop_rate=drop_rate,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  **kwargs).build_model()   


@MODELS.register("ResNet34")
def ResNet34(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
             activation=dict(activation="relu"),
             output_indices=(-1, ),
             strides=(2, 2, 2, 2, 2),
             dilation_rates=(1, 1, 1, 1, 1),
             frozen_stages=(-1, ),
             dropblock=None,
             num_classes=1000,
             drop_rate=0.5,
             input_shape=(224, 224, 3),
             input_tensor=None,
             **kwargs):
    return ResNet(name="resnet34",
                  blocks=[3, 4, 6, 3],
                  block_fn=BasicBlock,
                  normalization=normalization,
                  activation=activation,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  drop_rate=drop_rate,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  **kwargs).build_model()   


@MODELS.register("ResNet50")
def ResNet50(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
             activation=dict(activation="relu"),
             output_indices=(-1, ),
             strides=(2, 2, 2, 2, 2),
             dilation_rates=(1, 1, 1, 1, 1),
             frozen_stages=(-1, ),
             dropblock=None,
             num_classes=1000,
             drop_rate=0.5,
             input_shape=(224, 224, 3),
             input_tensor=None,
             **kwargs):

    return ResNet(name="resnet50",
                  blocks=[3, 4, 6, 3],
                  block_fn=Bottleneck,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@MODELS.register("ResNet101")
def ResNet101(convolution="conv2d",
              normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              output_indices=(-1, ),
              strides=(2, 2, 2, 2, 2),
              dilation_rates=(1, 1, 1, 1, 1),
              frozen_stages=(-1, ),
              dropblock=None,
              num_classes=1000,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    return ResNet(name="resnet101",
                  blocks=[3, 4, 23, 3],
                  block_fn=Bottleneck,
                  convolution=convolution,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


@MODELS.register("ResNet152")
def ResNet152(convolution="conv2d",
              normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              output_indices=(-1, ),
              strides=(2, 2, 2, 2, 2),
              dilation_rates=(1, 1, 1, 1, 1),
              frozen_stages=(-1, ),
              dropblock=None,
              num_classes=1000,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    
    return ResNet(name="resnet152",
                  blocks=[3, 8, 36, 3],
                  block_fn=Bottleneck,
                  convolution=convolution,
                  normalization=normalization,
                  activation=activation,
                  output_indices=output_indices,
                  strides=strides,
                  dilation_rates=dilation_rates,
                  frozen_stages=frozen_stages,
                  dropblock=dropblock,
                  num_classes=num_classes,
                  input_shape=input_shape,
                  input_tensor=input_tensor,
                  drop_rate=drop_rate,
                  **kwargs).build_model()


def _get_weight_name_map(blocks):
    name_map = {
        "conv1/conv2d/kernel:0": "conv1.weight",
        "conv1/batch_norm/gamma:0": "bn1.weight",
        "conv1/batch_norm/beta:0": "bn1.bias",
        "conv1/batch_norm/moving_mean:0": "bn1.running_mean",
        "conv1/batch_norm/moving_variance:0": "bn1.running_var",
    }

    for i in range(1, 5):
        for j in range(blocks[i - 1]):
            for k in range(1, 4):
                n1 = "layer%d/%d/conv%d" % (i, j, k)
                n2 = "layer%d.%d" % (i, j)
                m = {
                    "%s/conv2d/kernel:0" % n1: "%s.conv%d.weight" % (n2, k), 
                    "%s/batch_norm/gamma:0" % n1: "%s.bn%d.weight" % (n2, k),
                    "%s/batch_norm/beta:0" % n1: "%s.bn%d.bias" % (n2, k),
                    "%s/batch_norm/moving_mean:0" % n1: "%s.bn%d.running_mean" % (n2, k),
                    "%s/batch_norm/moving_variance:0" % n1: "%s.bn%d.running_var" % (n2, k),
                    "layer%d/0/shortcut/conv2d/kernel:0" % i: "layer%d.0.downsample.0.weight" % i,
                    "layer%d/0/shortcut/batch_norm/gamma:0" % i: "layer%d.0.downsample.1.weight" % i,
                    "layer%d/0/shortcut/batch_norm/beta:0" % i: "layer%d.0.downsample.1.bias" % i,
                    "layer%d/0/shortcut/batch_norm/moving_mean:0" % i: "layer%d.0.downsample.1.running_mean" % i,
                    "layer%d/0/shortcut/batch_norm/moving_variance:0" % i: "layer%d.0.downsample.1.running_var" % i
                }
                name_map.update(m)
    
    name_map["logits/kernel:0"] = "fc.weight"
    name_map["logits/bias:0"] = "fc.bias"

    return name_map


def _get_weights_from_pretrained(model, pretrained_weights_path, blocks):
    import torch
    import numpy as np

    pretrained = torch.load(pretrained_weights_path, map_location="cpu")
    # for k, v in pretrained.items():
    #     print(k)
    name_map = _get_weight_name_map(blocks)
    
    for w in model.weights:
        name = w.name
        pw = pretrained[name_map[name]].detach().numpy()
        if len(pw.shape) == 4:
            pw = np.transpose(pw, [2, 3, 1, 0])
        if len(pw.shape) == 2:
            pw = np.transpose(pw, [1, 0])
        w.assign(pw)


if __name__ == '__main__':
    from .common import fuse
    name = "resnet152"
    block_fn = Bottleneck
    blocks = [3, 8, 36, 3]
    resnet = ResNet(name=name,
                    blocks=blocks,
                    block_fn=block_fn,
                    normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True))
    
    model = resnet.build_model()
    model(tf.random.uniform([1, 224, 224, 3]))
    # model.summary()
    _get_weights_from_pretrained(model, "/Users/bailang/Downloads/pretrained_weights/%s.pth" % name, blocks)
    
    # fuse(model, block_fn)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    lbl = model(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)


