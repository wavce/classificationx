import tensorflow as tf
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from .common import DepthwiseConvNormActBlock
from core.layers import build_activation


class ChannelShuffle(tf.keras.layers.Layer):
    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)

        self.data_format = "channels_last" if data_format is None else data_format
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        inp_shape = tf.shape(inputs)
        if self.data_format == "channels_last":
            bs, h, w = inp_shape[0], inp_shape[1], inp_shape[2]
            chn = inputs.shape.as_list()[-1]
            inp = tf.reshape(inputs, [bs, h, w, chn // 2, 2])
            
            return inp[..., 0], inp[..., 1]

        bs, h, w = inp_shape[0], inp_shape[2], inp_shape[3]
        chn = inputs.shape.as_list()[1]
        inp = tf.reshape(inputs, [bs * chn // 2, 2, h * w])
        inp = tf.transpose(inp, [1, 0, 2])
        inp = tf.reshape(inp, [2, -1, chn // 2, h, w])

        return inp[0], inp[1]
    
    def get_config(self):
        base_config = super().get_config()
        base_config["data_format"] = self.data_format

        return base_config


def shuffle_v2_block(inputs,
                     in_filters,
                     out_filters,
                     kernel_size,
                     mid_filters,
                     strides=1,
                     groups=1,
                     kernel_initializer="he_normal",
                     data_format="channels_last",
                     normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                     activation=dict(activation="relu"),
                     trainable=True,
                     name="shuffle_v2_block"):
    out_filters -= in_filters
    if strides == 1:
        x1, x2 = ChannelShuffle(
            data_format=data_format,
            name=name + "/channel_shuffle")(inputs)
    else:
        x1 = DepthwiseConvNormActBlock(
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            use_bias="acon" in activation["activation"],
            normalization=normalization,
            activation=None,
            trainable=trainable,
            name=name + "/branch_proj/conv1")(inputs)
        if "acon" in activation["activation"]:
            activation["width"] = in_filters
        x1 = ConvNormActBlock(
            filters=in_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=kernel_initializer,
            use_bias="acon" in activation["activation"],
            normalization=normalization,
            activation=activation.copy(),
            trainable=trainable,
            name=name + "/branch_proj/conv2")(x1)
        x2 = inputs

    if "acon" in activation["activation"]:
        activation["width"] = mid_filters
    x2 = ConvNormActBlock(
        filters=mid_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=kernel_initializer,
        use_bias="acon" in activation["activation"],
        normalization=normalization,
        activation=activation.copy(),
        trainable=trainable,
        name=name + "/branch_main/conv1")(x2)
    x2 = DepthwiseConvNormActBlock(
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        use_bias="acon" in activation["activation"],
        normalization=normalization,
        activation=None,
        trainable=trainable,
        name=name + "/branch_main/conv2")(x2)
    if "acon" in activation["activation"]:
        activation["width"] = out_filters
    x2 = ConvNormActBlock(
        filters=out_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer=kernel_initializer,
        use_bias="acon" in activation["activation"],
        normalization=normalization,
        activation=activation.copy(),
        trainable=trainable,
        name=name + "/branch_main/conv3")(x2)
 
    axis = -1 if data_format == "channels_last" else 1
    return tf.keras.layers.Concatenate(axis=axis, name=name + "/concat")([x1, x2])


class ShuffleNetV2(Model):
    def __init__(self,
                 name,
                 blocks,
                 filters,
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
                 r=8,
                 **kwargs):
        super().__init__(
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
        self.r = r
    
    def build_model(self):
        if "acon" in self.activation["activation"]:
            self.activation["width"] = self.filters[1]
        if "meta" in self.activation["activation"]:
            self.activation["r"] = self.r
        x = ConvNormActBlock(
            filters=self.filters[1],
            kernel_size=3,
            strides=self.strides[0],
            normalization=self.normalization,
            activation=self.activation.copy(),
            data_format=self.data_format,
            use_bias="acon" in self.activation["activation"],
            name="first_conv")(self.img_input)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D(
            pool_size=3, 
            strides=self.strides[1],
            padding="valid", 
            data_format=self.data_format,
            name="maxpool")(x)
        
        block_idx = 0
        in_filters = self.filters[1]
        for stage_idx in range(len(self.blocks)):
            out_filters = self.filters[stage_idx + 2]
            for i in range(self.blocks[stage_idx]):
                if i == 0:
                    x = shuffle_v2_block(
                        inputs=x,
                        in_filters=in_filters,
                        out_filters=out_filters,
                        kernel_size=3,
                        mid_filters=out_filters // 2,
                        strides=self.strides[stage_idx + 2],
                        kernel_initializer=self.kernel_initializer,
                        data_format=self.data_format,
                        normalization=self.normalization,
                        activation=self.activation.copy(),
                        trainable=stage_idx + 1 not in self.frozen_stages,
                        name=str(block_idx))
                else:
                    x = shuffle_v2_block(
                        inputs=x,
                        in_filters=in_filters // 2,
                        out_filters=out_filters,
                        kernel_size=3,
                        mid_filters=out_filters // 2,
                        strides=1,
                        kernel_initializer=self.kernel_initializer,
                        data_format=self.data_format,
                        normalization=self.normalization,
                        activation=self.activation.copy(),
                        trainable=stage_idx + 1 not in self.frozen_stages,
                        name=str(block_idx))
                in_filters = out_filters
                block_idx += 1
        
        activation = self.activation.copy()
        if "acon" in activation["activation"]:
            activation["width"] = self.filters[-1]
        x = ConvNormActBlock(
            filters=self.filters[-1],
            kernel_size=1,
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            use_bias="acon" in self.activation["activation"],
            normalization=self.normalization,
            activation=activation,
            name="conv_last")(x)
        x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format)(x)
        if "2.0" in self.name:
            x = tf.keras.layers.Dropout(rate=self.drop_rate)(x)
        x = tf.keras.layers.Dense(
            units=self.num_classes, 
            use_bias="acon" in self.activation["activation"])(x)

        return tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name)


@MODELS.register("ShuffleNetV2_5X")
def ShuffleNetV2_5X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                    activation=dict(activation="relu"),
                    strides=(2, 2, 2, 2, 2),
                    dilation_rates=(1, 1, 1, 1, 1),
                    frozen_stages=(-1, ),
                    num_classes=512,
                    drop_rate=0.5,
                    input_shape=(224, 224, 3),
                    input_tensor=None,
                    **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_0.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 48, 96, 192, 1024],
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


@MODELS.register("ShuffleNetV2_10X")
def ShuffleNetV2_10X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                     activation=dict(activation="relu"),
                     strides=(2, 2, 2, 2, 2),
                     dilation_rates=(1, 1, 1, 1, 1),
                     frozen_stages=(-1, ),
                     num_classes=512,
                     drop_rate=0.5,
                     input_shape=(224, 224, 3),
                     input_tensor=None,
                     **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_1.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 116, 232, 464, 1024],
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


@MODELS.register("ShuffleNetV2_15X")
def ShuffleNetV2_15X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                     activation=dict(activation="relu"),
                     strides=(2, 2, 2, 2, 2),
                     dilation_rates=(1, 1, 1, 1, 1),
                     frozen_stages=(-1, ),
                     num_classes=512,
                     drop_rate=0.5,
                     input_shape=(224, 224, 3),
                     input_tensor=None,
                     **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_1.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 176, 352, 704, 1024],
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


@MODELS.register("ShuffleNetV2_20X")
def ShuffleNetV2_20X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                     activation=dict(activation="relu"),
                     strides=(2, 2, 2, 2, 2),
                     dilation_rates=(1, 1, 1, 1, 1),
                     frozen_stages=(-1, ),
                     num_classes=512,
                     drop_rate=0.5,
                     input_shape=(224, 224, 3),
                     input_tensor=None,
                     **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_2.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 244, 488, 976, 2048],
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


@MODELS.register("ShuffleNetV2ACON5X")
def ShuffleNetV2ACON5X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                       strides=(2, 2, 2, 2, 2),
                       dilation_rates=(1, 1, 1, 1, 1),
                       frozen_stages=(-1, ),
                       num_classes=512,
                       drop_rate=0.5,
                       input_shape=(224, 224, 3),
                       input_tensor=None,
                       **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_acon_0.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 48, 96, 192, 1024],
        normalization=normalization,
        activation=dict(activation="aconc"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV2ACON10X")
def ShuffleNetV2ACON10X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                        strides=(2, 2, 2, 2, 2),
                        dilation_rates=(1, 1, 1, 1, 1),
                        frozen_stages=(-1, ),
                        num_classes=512,
                        drop_rate=0.5,
                        input_shape=(224, 224, 3),
                        input_tensor=None,
                        **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_acon_1.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 116, 232, 464, 1024],
        normalization=normalization,
        activation=dict(activation="aconc"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV2ACON15X")
def ShuffleNetV2ACON15X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                        strides=(2, 2, 2, 2, 2),
                        dilation_rates=(1, 1, 1, 1, 1),
                        frozen_stages=(-1, ),
                        num_classes=512,
                        drop_rate=0.5,
                        input_shape=(224, 224, 3),
                        input_tensor=None,
                        **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_acon_1.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 176, 352, 704, 1024],
        normalization=normalization,
        activation=dict(activation="aconc"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model() 


@MODELS.register("ShuffleNetV2ACON20X")
def ShuffleNetV2ACON20X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                        strides=(2, 2, 2, 2, 2),
                        dilation_rates=(1, 1, 1, 1, 1),
                        frozen_stages=(-1, ),
                        num_classes=512,
                        drop_rate=0.5,
                        input_shape=(224, 224, 3),
                        input_tensor=None,
                        **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_acon_2.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 244, 488, 976, 2048],
        normalization=normalization,
        activation=dict(activation="aconc"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model() 


@MODELS.register("ShuffleNetV2MetaACON5X")
def ShuffleNetV2MetaACON5X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                           strides=(2, 2, 2, 2, 2),
                           dilation_rates=(1, 1, 1, 1, 1),
                           frozen_stages=(-1, ),
                           num_classes=512,
                           drop_rate=0.5,
                           input_shape=(224, 224, 3),
                           input_tensor=None,
                           **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_meta_acon_0.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 48, 96, 192, 1024],
        normalization=normalization,
        activation=dict(activation="meta_acon"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        r=8
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV2MetaACON10X")
def ShuffleNetV2MetaACON10X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=512,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_meta_acon_1.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 116, 232, 464, 1024],
        normalization=normalization,
        activation=dict(activation="meta_acon"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        r=8,
        **kwargs).build_model()   


@MODELS.register("ShuffleNetV2MetaACON15X")
def ShuffleNetV2MetaACON15X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=512,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_meta_acon_1.5x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 176, 352, 704, 1024],
        normalization=normalization,
        activation=dict(activation="meta_acon"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        r=16,
        **kwargs).build_model() 


@MODELS.register("ShuffleNetV2MetaACON20X")
def ShuffleNetV2MetaACON20X(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                            strides=(2, 2, 2, 2, 2),
                            dilation_rates=(1, 1, 1, 1, 1),
                            frozen_stages=(-1, ),
                            num_classes=512,
                            drop_rate=0.5,
                            input_shape=(224, 224, 3),
                            input_tensor=None,
                            **kwargs):
    return ShuffleNetV2(
        name="shufflenet_v2_meta_acon_2.0x",
        blocks=[4, 8, 4],
        filters=[-1, 24, 244, 488, 976, 2048],
        normalization=normalization,
        activation=dict(activation="meta_acon"),
        strides=strides,
        dilation_rates=dilation_rates,
        frozen_stages=frozen_stages,
        num_classes=num_classes,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        r=16,
        **kwargs).build_model() 


def _get_weight_name_map(blocks):
    name_map = {
        "first_conv/conv2d/kernel:0"               : "module.first_conv.0.weight",
        "first_conv/conv2d/bias:0"                 : "module.first_conv.0.bias",
        "first_conv/batch_norm/gamma:0"            : "module.first_conv.1.weight",
        "first_conv/batch_norm/beta:0"             : "module.first_conv.1.bias",
        "first_conv/batch_norm/moving_mean:0"      : "module.first_conv.1.running_mean",
        "first_conv/batch_norm/moving_variance:0"  : "module.first_conv.1.running_var",
        "first_conv/aconc/p1:0"                    : "module.first_conv.2.p1",
        "first_conv/aconc/p2:0"                    : "module.first_conv.2.p2",
        "first_conv/aconc/beta:0"                  : "module.first_conv.2.beta",
        "first_conv/aconc/fc1/kernel:0"            : "module.first_conv.2.fc1.weight",
        "first_conv/aconc/fc1/bias:0"              : "module.first_conv.2.fc1.bias",
        "first_conv/aconc/norm1/gamma:0"           : "module.first_conv.2.bn1.weight",
        "first_conv/aconc/norm1/beta:0"            : "module.first_conv.2.bn1.bias",
        "first_conv/aconc/norm1/moving_mean:0"     : "module.first_conv.2.bn1.running_mean",
        "first_conv/aconc/norm1/moving_variance:0" : "module.first_conv.2.bn1.running_var",
        "first_conv/aconc/fc2/kernel:0"            : "module.first_conv.2.fc2.weight",
        "first_conv/aconc/fc2/bias:0"              : "module.first_conv.2.fc2.bias",
        "first_conv/aconc/norm2/gamma:0"           : "module.first_conv.2.bn2.weight",
        "first_conv/aconc/norm2/beta:0"            : "module.first_conv.2.bn2.bias",
        "first_conv/aconc/norm2/moving_mean:0"     : "module.first_conv.2.bn2.running_mean",
        "first_conv/aconc/norm2/moving_variance:0" : "module.first_conv.2.bn2.running_var",
    }

    idx = 0
    for n in blocks:
        for i in range(n): 
            m = {
                "%d/branch_main/conv1/conv2d/kernel:0" % idx               : "module.features.%d.branch_main.0.weight" % idx,
                "%d/branch_main/conv1/conv2d/bias:0" % idx                 : "module.features.%d.branch_main.0.bias" % idx,
                "%d/branch_main/conv1/batch_norm/gamma:0" % idx            : "module.features.%d.branch_main.1.weight" % idx,
                "%d/branch_main/conv1/batch_norm/beta:0" % idx             : "module.features.%d.branch_main.1.bias" % idx,
                "%d/branch_main/conv1/batch_norm/moving_mean:0" % idx      : "module.features.%d.branch_main.1.running_mean" % idx,
                "%d/branch_main/conv1/batch_norm/moving_variance:0" % idx  : "module.features.%d.branch_main.1.running_var" % idx,
                "%d/branch_main/conv1/aconc/p1:0" % idx                    : "module.features.%d.branch_main.2.p1" % idx,
                "%d/branch_main/conv1/aconc/p2:0" % idx                    : "module.features.%d.branch_main.2.p2" % idx,
                "%d/branch_main/conv1/aconc/beta:0" % idx                  : "module.features.%d.branch_main.2.beta" % idx,
                "%d/branch_main/conv1/aconc/fc1/kernel:0" % idx            : "module.features.%d.branch_main.2.fc1.weight" % idx,
                "%d/branch_main/conv1/aconc/fc1/bias:0" % idx              : "module.features.%d.branch_main.2.fc1.bias" % idx,
                "%d/branch_main/conv1/aconc/norm1/gamma:0" % idx           : "module.features.%d.branch_main.2.bn1.weight" % idx,
                "%d/branch_main/conv1/aconc/norm1/beta:0" % idx            : "module.features.%d.branch_main.2.bn1.bias" % idx,
                "%d/branch_main/conv1/aconc/norm1/moving_mean:0" % idx     : "module.features.%d.branch_main.2.bn1.running_mean" % idx,
                "%d/branch_main/conv1/aconc/norm1/moving_variance:0" % idx : "module.features.%d.branch_main.2.bn1.running_var" % idx,
                "%d/branch_main/conv1/aconc/fc2/kernel:0" % idx            : "module.features.%d.branch_main.2.fc2.weight" % idx,
                "%d/branch_main/conv1/aconc/fc2/bias:0" % idx              : "module.features.%d.branch_main.2.fc2.bias" % idx,
                "%d/branch_main/conv1/aconc/norm2/gamma:0" % idx           : "module.features.%d.branch_main.2.bn2.weight" % idx,
                "%d/branch_main/conv1/aconc/norm2/beta:0" % idx            : "module.features.%d.branch_main.2.bn2.bias" % idx,
                "%d/branch_main/conv1/aconc/norm2/moving_mean:0" % idx     : "module.features.%d.branch_main.2.bn2.running_mean" % idx,
                "%d/branch_main/conv1/aconc/norm2/moving_variance:0" % idx : "module.features.%d.branch_main.2.bn2.running_var" % idx,
                "%d/branch_main/conv2/conv2d/depthwise_kernel:0" % idx     : "module.features.%d.branch_main.3.weight" % idx,
                "%d/branch_main/conv2/conv2d/bias:0" % idx                 : "module.features.%d.branch_main.3.bias" % idx,
                "%d/branch_main/conv2/batch_norm/gamma:0" % idx            : "module.features.%d.branch_main.4.weight" % idx,
                "%d/branch_main/conv2/batch_norm/beta:0" % idx             : "module.features.%d.branch_main.4.bias" % idx,
                "%d/branch_main/conv2/batch_norm/moving_mean:0" % idx      : "module.features.%d.branch_main.4.running_mean" % idx,
                "%d/branch_main/conv2/batch_norm/moving_variance:0" % idx  : "module.features.%d.branch_main.4.running_var" % idx,
                "%d/branch_main/conv3/conv2d/kernel:0" % idx               : "module.features.%d.branch_main.5.weight" % idx,
                "%d/branch_main/conv3/conv2d/bias:0" % idx                 : "module.features.%d.branch_main.5.bias" % idx,
                "%d/branch_main/conv3/batch_norm/gamma:0" % idx            : "module.features.%d.branch_main.6.weight" % idx,
                "%d/branch_main/conv3/batch_norm/beta:0" % idx             : "module.features.%d.branch_main.6.bias" % idx,
                "%d/branch_main/conv3/batch_norm/moving_mean:0" % idx      : "module.features.%d.branch_main.6.running_mean" % idx,
                "%d/branch_main/conv3/batch_norm/moving_variance:0" % idx  : "module.features.%d.branch_main.6.running_var" % idx,
                "%d/branch_main/conv3/aconc/p1:0" % idx                    : "module.features.%d.branch_main.7.p1" % idx,
                "%d/branch_main/conv3/aconc/p2:0" % idx                    : "module.features.%d.branch_main.7.p2" % idx,
                "%d/branch_main/conv3/aconc/beta:0" % idx                  : "module.features.%d.branch_main.7.beta" % idx,
                "%d/branch_main/conv3/aconc/fc1/kernel:0" % idx            : "module.features.%d.branch_main.7.fc1.weight" % idx,
                "%d/branch_main/conv3/aconc/fc1/bias:0" % idx              : "module.features.%d.branch_main.7.fc1.bias" % idx,
                "%d/branch_main/conv3/aconc/norm1/gamma:0" % idx           : "module.features.%d.branch_main.7.bn1.weight" % idx,
                "%d/branch_main/conv3/aconc/norm1/beta:0" % idx            : "module.features.%d.branch_main.7.bn1.bias" % idx,
                "%d/branch_main/conv3/aconc/norm1/moving_mean:0" % idx     : "module.features.%d.branch_main.7.bn1.running_mean" % idx,
                "%d/branch_main/conv3/aconc/norm1/moving_variance:0" % idx : "module.features.%d.branch_main.7.bn1.running_var" % idx,
                "%d/branch_main/conv3/aconc/fc2/kernel:0" % idx            : "module.features.%d.branch_main.7.fc2.weight" % idx,
                "%d/branch_main/conv3/aconc/fc2/bias:0" % idx              : "module.features.%d.branch_main.7.fc2.bias" % idx,
                "%d/branch_main/conv3/aconc/norm2/gamma:0" % idx           : "module.features.%d.branch_main.7.bn2.weight" % idx,
                "%d/branch_main/conv3/aconc/norm2/beta:0" % idx            : "module.features.%d.branch_main.7.bn2.bias" % idx,
                "%d/branch_main/conv3/aconc/norm2/moving_mean:0" % idx     : "module.features.%d.branch_main.7.bn2.running_mean" % idx,
                "%d/branch_main/conv3/aconc/norm2/moving_variance:0" % idx : "module.features.%d.branch_main.7.bn2.running_var" % idx,
            }
            name_map.update(m)
            if i == 0:
                m = {
                    "%d/branch_proj/conv1/conv2d/depthwise_kernel:0" % idx     : "module.features.%d.branch_proj.0.weight" % idx,
                    "%d/branch_proj/conv1/conv2d/bias:0" % idx                 : "module.features.%d.branch_proj.0.bias" % idx,
                    "%d/branch_proj/conv1/batch_norm/gamma:0" % idx            : "module.features.%d.branch_proj.1.weight" % idx,
                    "%d/branch_proj/conv1/batch_norm/beta:0" % idx             : "module.features.%d.branch_proj.1.bias" % idx,
                    "%d/branch_proj/conv1/batch_norm/moving_mean:0" % idx      : "module.features.%d.branch_proj.1.running_mean" % idx,
                    "%d/branch_proj/conv1/batch_norm/moving_variance:0" % idx  : "module.features.%d.branch_proj.1.running_var" % idx,
                    "%d/branch_proj/conv2/conv2d/kernel:0" % idx               : "module.features.%d.branch_proj.2.weight" % idx,
                    "%d/branch_proj/conv2/conv2d/bias:0" % idx                 : "module.features.%d.branch_proj.2.bias" % idx,
                    "%d/branch_proj/conv2/batch_norm/gamma:0" % idx            : "module.features.%d.branch_proj.3.weight" % idx,
                    "%d/branch_proj/conv2/batch_norm/beta:0" % idx             : "module.features.%d.branch_proj.3.bias" % idx,
                    "%d/branch_proj/conv2/batch_norm/moving_mean:0" % idx      : "module.features.%d.branch_proj.3.running_mean" % idx,
                    "%d/branch_proj/conv2/batch_norm/moving_variance:0" % idx  : "module.features.%d.branch_proj.3.running_var" % idx,
                    "%d/branch_proj/conv2/aconc/p1:0" % idx                    : "module.features.%d.branch_proj.4.p1" % idx,
                    "%d/branch_proj/conv2/aconc/p2:0" % idx                    : "module.features.%d.branch_proj.4.p2" % idx,
                    "%d/branch_proj/conv2/aconc/beta:0" % idx                  : "module.features.%d.branch_proj.4.beta" % idx,
                    "%d/branch_proj/conv2/aconc/fc1/kernel:0" % idx            : "module.features.%d.branch_proj.4.fc1.weight" % idx,
                    "%d/branch_proj/conv2/aconc/fc1/bias:0" % idx              : "module.features.%d.branch_proj.4.fc1.bias" % idx,
                    "%d/branch_proj/conv2/aconc/norm1/gamma:0" % idx           : "module.features.%d.branch_proj.4.bn1.weight" % idx,
                    "%d/branch_proj/conv2/aconc/norm1/beta:0" % idx            : "module.features.%d.branch_proj.4.bn1.bias" % idx,
                    "%d/branch_proj/conv2/aconc/norm1/moving_mean:0" % idx     : "module.features.%d.branch_proj.4.bn1.running_mean" % idx,
                    "%d/branch_proj/conv2/aconc/norm1/moving_variance:0" % idx : "module.features.%d.branch_proj.4.bn1.running_var" % idx,
                    "%d/branch_proj/conv2/aconc/fc2/kernel:0" % idx            : "module.features.%d.branch_proj.4.fc2.weight" % idx,
                    "%d/branch_proj/conv2/aconc/fc2/bias:0" % idx              : "module.features.%d.branch_proj.4.fc2.bias" % idx,
                    "%d/branch_proj/conv2/aconc/norm2/gamma:0" % idx           : "module.features.%d.branch_proj.4.bn2.weight" % idx,
                    "%d/branch_proj/conv2/aconc/norm2/beta:0" % idx            : "module.features.%d.branch_proj.4.bn2.bias" % idx,
                    "%d/branch_proj/conv2/aconc/norm2/moving_mean:0" % idx     : "module.features.%d.branch_proj.4.bn2.running_mean" % idx,
                    "%d/branch_proj/conv2/aconc/norm2/moving_variance:0" % idx : "module.features.%d.branch_proj.4.bn2.running_var" % idx,
                }
                name_map.update(m)
            idx += 1

    name_map["conv_last/conv2d/kernel:0"]                = "module.conv_last.0.weight"
    name_map["conv_last/conv2d/bias:0"]                  = "module.conv_last.0.bias"
    name_map["conv_last/batch_norm/gamma:0"]             = "module.conv_last.1.weight"
    name_map["conv_last/batch_norm/beta:0"]              = "module.conv_last.1.bias"
    name_map["conv_last/batch_norm/moving_mean:0"]       = "module.conv_last.1.running_mean"
    name_map["conv_last/batch_norm/moving_variance:0"]   = "module.conv_last.1.running_var"
    name_map["conv_last/aconc/p1:0"]                     = "module.conv_last.2.p1"
    name_map["conv_last/aconc/p2:0"]                     = "module.conv_last.2.p2"
    name_map["conv_last/aconc/beta:0"]                   = "module.conv_last.2.beta"
    name_map["conv_last/aconc/fc1/kernel:0"]             = "module.conv_last.2.fc1.weight"
    name_map["conv_last/aconc/fc1/bias:0"]               = "module.conv_last.2.fc1.bias"
    name_map["conv_last/aconc/norm1/gamma:0"]            = "module.conv_last.2.bn1.weight"
    name_map["conv_last/aconc/norm1/beta:0"]             = "module.conv_last.2.bn1.bias"
    name_map["conv_last/aconc/norm1/moving_mean:0"]      = "module.conv_last.2.bn1.running_mean"
    name_map["conv_last/aconc/norm1/moving_variance:0"]  = "module.conv_last.2.bn1.running_var"
    name_map["conv_last/aconc/fc2/kernel:0"]             = "module.conv_last.2.fc2.weight"
    name_map["conv_last/aconc/fc2/bias:0"]               = "module.conv_last.2.fc2.bias"
    name_map["conv_last/aconc/norm2/gamma:0"]            = "module.conv_last.2.bn2.weight"
    name_map["conv_last/aconc/norm2/beta:0"]             = "module.conv_last.2.bn2.bias"
    name_map["conv_last/aconc/norm2/moving_mean:0"]      = "module.conv_last.2.bn2.running_mean"
    name_map["conv_last/aconc/norm2/moving_variance:0"]  = "module.conv_last.2.bn2.running_var"
    name_map["dense/kernel:0"]                           = "module.classifier.0.weight"
    name_map["dense/bias:0"]                             = "module.classifier.0.bias"

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
        if "meta_acon" in name:
            name = name.replace("meta_acon", "aconc")
        # print(name, w.shape.as_list())
        pw = pretrained[name_map[name]].detach().numpy()
        if len(pw.shape) == 4 and "depthwise_kernel" not in name:
            pw = np.transpose(pw, [2, 3, 1, 0])
        if "depthwise_kernel" in name:
            pw = np.transpose(pw, [2, 3, 0, 1])
        if "aconc" in name and "fc" not in name:
            pw = np.squeeze(pw)
        if len(pw.shape) == 2:
            pw = np.transpose(pw, [1, 0])
        
        w.assign(pw)


if __name__ == '__main__':
    # from ..common import fuse
    name = "shufflenet_v2_1.5x_meta_acon"
    
    model = ShuffleNetV2MetaACON15X(
        input_shape=(224, 224, 3),
        num_classes=1000)
    # model.summary()
    _get_weights_from_pretrained(
        model, 
        "/Users/bailang/Downloads/meta-acon/%s.pth" % name, 
        blocks=[4, 8, 4])
    
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
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s.h5" % name)
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/%s/model.ckpt" % name)
    