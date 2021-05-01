import tensorflow as tf
from .model import Model
from .builder import MODELS 
from .common import ConvNormActBlock
from .common import DepthwiseConvNormActBlock
from core.layers import build_activation


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, groups, data_format=None, **kwargs):
        super().__init__(**kwargs)

        self.groups = groups

        self.data_format = "channels_last" if data_format is None else data_format
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        inp_shape = tf.shape(inputs)
        if self.data_format == "channels_last":
            bs, h, w, chn = inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]
            inp = tf.reshape(inputs, [bs, h, w, chn // self.groups, self.groups])
            inp = tf.transpose(inp, [0, 1, 2, 4, 3])
        else:
            bs, chn, h, w = inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[3]
            inp = tf.reshape(inputs, [bs, chn // self.groups, self.groups, h, w])
            inp = tf.transpose(inp, [0, 2, 1, 3, 4])

        inp = tf.reshape(inp, inp_shape)
        return inp
    
    def get_config(self):
        base_config = super().get_config()
        base_config["groups"] = self.groups

        return base_config


def shuffle_v1_block(inputs,
                     in_filters,
                     out_filters,
                     kernel_size,
                     mid_filters,
                     strides=1,
                     groups=1,
                     first_group=False,
                     kernel_initializer="he_normal",
                     data_format="channels_last",
                     normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                     activation=dict(activation="relu"),
                     trainable=True,
                     name="shuffle_v1_block"):
 
    x = ConvNormActBlock(
        filters=mid_filters,
        kernel_size=1,
        strides=1,
        groups=1 if first_group else groups,
        kernel_initializer=kernel_initializer,
        normalization=normalization,
        activation=activation,
        trainable=trainable,
        name=name + "/conv1")(inputs)
    x = DepthwiseConvNormActBlock(
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        normalization=normalization,
        activation=None,
        trainable=trainable,
        name=name + "/conv2")(x)
    if groups > 1:
        x = ChannelShuffle(
            groups=groups,
            data_format=data_format,
            name=name + "/channel_shuffle")(x)
    x = ConvNormActBlock(
        filters=out_filters - in_filters if strides == 2 else out_filters,
        kernel_size=1,
        strides=1,
        groups=groups,
        kernel_initializer=kernel_initializer,
        normalization=normalization,
        activation=None,
        trainable=trainable,
        name=name + "/conv3")(x)
    
    if strides == 1:
        x = tf.keras.layers.Add(name=name + "/sum")([x, inputs])
        return x
    
    inputs = tf.keras.layers.AvgPool2D(
        pool_size=3,
        strides=strides,
        padding="same",
        data_format=data_format,
        name=name + "/avgpool")(inputs)
    x = build_activation(
        name=name + "/%s" % activation["activation"], 
        **activation)(x)
    
    axis = -1 if data_format == "channels_last" else 1
    return tf.keras.layers.Concatenate(axis=axis, name=name + "/concat")([inputs, x])


class ShuffleNetV1(Model):
    def __init__(self,
                 name,
                 blocks,
                 filters,
                 groups,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                 strides=(2, 2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1, 1),
                 frozen_stages=(-1, ),
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.5,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 **kwargs):
        super(ShuffleNetV1, self).__init__(
            name=name,
            kernel_initializer=kernel_initializer,
            normalization=normalization,
            activation=activation,
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
        self.filters = filters
        self.groups = groups
    
    def build_model(self):
        x = ConvNormActBlock(
            filters=self.filters[1],
            kernel_size=3,
            strides=self.strides[0],
            normalization=self.normalization,
            activation=self.activation,
            data_format=self.data_format,
            name="first_conv")(self.img_input)
        
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=3, 
            strides=self.strides[1],
            padding="valid", 
            data_format=self.data_format,
            name="maxpool")(x)
        
        block_idx = 0
        for stage_idx in range(len(self.blocks)):
            for i in range(self.blocks[stage_idx]):
                strides = self.strides[stage_idx + 1] if i == 0 else 1
                x = shuffle_v1_block(
                    inputs=x,
                    in_filters=self.filters[stage_idx + 1],
                    out_filters=self.filters[stage_idx + 2],
                    kernel_size=3,
                    mid_filters=self.filters[stage_idx + 2] // 4,
                    strides=strides,
                    groups=self.groups,
                    first_group=i == 0 and stage_idx == 0,
                    kernel_initializer=self.kernel_initializer,
                    data_format=self.data_format,
                    normalization=self.normalization,
                    activation=self.activation,
                    trainable=stage_idx + 1 not in self.frozen_stages,
                    name=str(block_idx))
                block_idx += 1
        x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
        x = tf.keras.layers.Dense(units=self.num_classes, use_bias=False)(x)

        return tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name)


@MODELS.register("ShuffleNetV1_Group3_5X")
def ShuffleNetV1_Group3_5X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                           activation=dict(activation="relu"),
                           strides=(2, 2, 2, 2, 2),
                           dilation_rates=(1, 1, 1, 1, 1),
                           frozen_stages=(-1, ),
                           num_classes=1000,
                           drop_rate=0.5,
                           input_shape=(224, 224, 3),
                           input_tensor=None,
                           **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group3_0.5x",
        blocks=[4, 8, 4],
        groups=3,
        filters=[-1, 12, 120, 240, 480],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group3_10X")
def ShuffleNetV1_Group3_10X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group3_1.0x",
        blocks=[4, 8, 4],
        groups=3,
        filters=[-1, 24, 240, 480, 960],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group3_15X")
def ShuffleNetV1_Group3_15X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group3_1.5x",
        blocks=[4, 8, 4],
        groups=3,
        filters=[-1, 24, 360, 720, 1440],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group3_20X")
def ShuffleNetV1_Group3_20X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group3_2.0x",
        blocks=[4, 8, 4],
        groups=3,
        filters=[-1, 48, 480, 960, 1920],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group8_5X")
def ShuffleNetV1_Group8_5X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                           activation=dict(activation="relu"),
                           strides=(2, 2, 2, 2, 2),
                           dilation_rates=(1, 1, 1, 1, 1),
                           frozen_stages=(-1, ),
                           num_classes=1000,
                           drop_rate=0.5,
                           input_shape=(224, 224, 3),
                           input_tensor=None,
                           **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group8_0.5x",
        blocks=[4, 8, 4],
        groups=8,
        filters=[-1, 16, 192, 384, 768],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group8_10X")
def ShuffleNetV1_Group8_10X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group8_1.0x",
        blocks=[4, 8, 4],
        groups=8,
        filters=[-1, 24, 384, 768, 1536],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group8_15X")
def ShuffleNetV1_Group8_15X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group8_1.5x",
        blocks=[4, 8, 4],
        groups=8,
        filters=[-1, 24, 576, 1152, 2304],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV1_Group8_20X")
def ShuffleNetV1_Group8_20X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            activation=dict(activation="relu"),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=1000,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV1(
        name="shufflenet_v1_group8_2.0x",
        blocks=[4, 8, 4],
        groups=8,
        filters=[-1, 48, 768, 1536, 3072],
        normalization=normalization,
        activation=activation,
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   



def _get_weight_name_map(blocks):
    name_map = {
        "first_conv/conv2d/kernel:0":               "module.first_conv.0.weight",
        "first_conv/batch_norm/gamma:0":            "module.first_conv.1.weight",
        "first_conv/batch_norm/beta:0":             "module.first_conv.1.bias",
        "first_conv/batch_norm/moving_mean:0":      "module.first_conv.1.running_mean",
        "first_conv/batch_norm/moving_variance:0":  "module.first_conv.1.running_var",
    }

    idx = 0
    for n in blocks:
        for _ in range(n): 
            m = {
                "%d/conv1/conv2d/kernel:0" % idx :               "module.features.%d.branch_main_1.0.weight" % idx,
                "%d/conv1/batch_norm/gamma:0" % idx:             "module.features.%d.branch_main_1.1.weight" % idx,
                "%d/conv1/batch_norm/beta:0" % idx:              "module.features.%d.branch_main_1.1.bias" % idx,
                "%d/conv1/batch_norm/moving_mean:0" % idx:       "module.features.%d.branch_main_1.1.running_mean" % idx,
                "%d/conv1/batch_norm/moving_variance:0" % idx:   "module.features.%d.branch_main_1.1.running_var" % idx,
                "%d/conv2/conv2d/depthwise_kernel:0" % idx:      "module.features.%d.branch_main_1.3.weight" % idx,
                "%d/conv2/batch_norm/gamma:0" % idx:             "module.features.%d.branch_main_1.4.weight" % idx,
                "%d/conv2/batch_norm/beta:0" % idx:              "module.features.%d.branch_main_1.4.bias" % idx,
                "%d/conv2/batch_norm/moving_mean:0" % idx:       "module.features.%d.branch_main_1.4.running_mean" % idx,
                "%d/conv2/batch_norm/moving_variance:0" % idx:   "module.features.%d.branch_main_1.4.running_var" % idx,
                "%d/conv3/conv2d/kernel:0" % idx:                "module.features.%d.branch_main_2.0.weight" % idx,
                "%d/conv3/batch_norm/gamma:0" % idx:             "module.features.%d.branch_main_2.1.weight" % idx,
                "%d/conv3/batch_norm/beta:0" % idx:              "module.features.%d.branch_main_2.1.bias" % idx,
                "%d/conv3/batch_norm/moving_mean:0" % idx:       "module.features.%d.branch_main_2.1.running_mean" % idx,
                "%d/conv3/batch_norm/moving_variance:0" % idx:   "module.features.%d.branch_main_2.1.running_var" % idx,
            }
            name_map.update(m)
            idx += 1

    name_map["dense/kernel:0"] = "module.classifier.0.weight"

    return name_map


def _get_weights_from_pretrained(model, pretrained_weights_path, blocks):
    import torch
    import numpy as np

    pretrained = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]
    # for k, v in pretrained.items():
    #     if "tracked" not in k:
    #         print(k, v.numpy().shape)
    name_map = _get_weight_name_map(blocks)
    
    for w in model.weights:
        name = w.name
        print(name, w.shape.as_list())
        pw = pretrained[name_map[name]].detach().numpy()
        if len(pw.shape) == 4 and "depthwise_kernel" not in name:
            pw = np.transpose(pw, [2, 3, 1, 0])
        if "depthwise_kernel" in name:
            pw = np.transpose(pw, [2, 3, 0, 1])
        if len(pw.shape) == 2:
            pw = np.transpose(pw, [1, 0])
        w.assign(pw)


if __name__ == '__main__':
    # from ..common import fuse
    name = "shufflenet_v1_group8_0.5x"
    
    model = ShuffleNetV1_Group8_5X(
        input_shape=(224, 224, 3),
        num_classes=1000)
    # model.summary()
    _get_weights_from_pretrained(model, "/Users/bailang/Downloads/ShuffleNetV1/Group8/%s.pth.tar" % name, blocks=[4, 8, 4])
    
    # fuse(model, block_fn)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))
    images = tf.expand_dims(images, axis=0)
    lbl = model(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    # tf.saved_model.save( model, )
    # model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    # model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)
