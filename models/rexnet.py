import math
import tensorflow as tf
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from .common import DepthwiseConvNormActBlock
from core.layers import build_activation
from .common import squeeze_excitation


class Shortcut(tf.keras.layers.Layer):
    def __init__(self, in_filters, data_format="channels_last", **kwargs):
        super().__init__(**kwargs)

        self.in_filters = in_filters
        self.data_format = data_format
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs, shortcut):
        input_shape = tf.shape(inputs)
        if self.data_format == "channels_last":
            x = inputs[..., :self.in_filters] + shortcut
            x = tf.concat([x, inputs[..., self.in_filters:]], axis=-1)
        else:
            x = inputs[:, :self.in_filters] + shortcut
            x = tf.concat([x, inputs[:, self.in_filters:]], axis=1)
        
        x = tf.reshape(x, input_shape)

        return x
    
    def get_config(self):
        base_config = super().get_config()
        base_config["data_format"] = self.data_format
        base_config["in_filters"] = self.in_filters

        return base_config


def linear_bottleneck_v1(inputs,
                         t,
                         in_filters,
                         out_filters,
                         strides=1,
                         use_se=True, 
                         se_ratio=12,
                         kernel_initializer="he_normal",
                         data_format="channels_last",
                         normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                         trainable=True,
                         name="linear_bottleneck"):
    use_shorcut = strides == 1 and in_filters <= out_filters

    x = inputs
    if t != 1:
        dw_filters = in_filters * t
        x = ConvNormActBlock(
            filters=dw_filters,
            kernel_size=1,
            normalization=normalization,
            activation=dict(activation="swish"),
            data_format=data_format,
            kernel_initializer=kernel_initializer,
            trainable=trainable,
            name=name + "/conv1")(x)
    else:
        dw_filters = in_filters
    x = DepthwiseConvNormActBlock(
        kernel_size=3,
        strides=strides,
        data_format=data_format,
        kernel_initializer=kernel_initializer,
        normalization=normalization,
        name=name + "/conv2")(x)
    if use_se:
        x = squeeze_excitation(
            inputs=x, 
            in_filters=dw_filters, 
            se_ratio=se_ratio, 
            data_format=data_format, 
            trainable=trainable, 
            normalization=normalization,
            name=name + "/se")
    
    x = build_activation(activation="relu6", name=name + "/relu6")(x)
    x = ConvNormActBlock(
        filters=out_filters,
        kernel_size=1,
        data_format=data_format,
        kernel_initializer=kernel_initializer,
        trainable=trainable,
        normalization=normalization,
        activation=None,
        name=name + "/conv3")(x)
    
    if use_shorcut:
        x = Shortcut(
            in_filters=in_filters,
            data_format=data_format,
            name=name + "/shortcut")(x, inputs)
    
    return x


class ReXNet(Model):
    def __init__(self,
                 name,
                 use_se=True,
                 se_ratio=12,
                 input_filters=16,
                 final_filters=180,
                 width_multiplier=1.0, 
                 depth_multiplier=1.0,
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 kernel_initializer=tf.keras.initializers.VarianceScaling(),
                 strides=[1, 2, 2, 2, 1, 2],
                 dilation_rates=(1, 1, 1, 1, 1, 1),
                 dropblock=None,
                 num_classes=1000,
                 drop_rate=0.2,
                 input_shape=(224, 224, 3),
                 input_tensor=None,
                 **kwargs):
        super().__init__(
            name=name,
            kernel_initializer=kernel_initializer,
            normalization=normalization,
            activation=activation,
            strides=strides,
            dilation_rates=dilation_rates,
            frozen_stages=None,
            dropblock=dropblock,
            num_classes=num_classes,
            input_shape=input_shape,
            input_tensor=input_tensor,
            drop_rate=drop_rate,
            **kwargs)
        layers = [1, 2, 2, 3, 3, 5]
        use_ses = [False, False, True, True, True, True]

        self.layers = [math.ceil(element * depth_multiplier) for element in layers]
        self.strides = sum(
            [[element] + [1] * (layers[idx] - 1) for idx, element in enumerate(strides)], [])
        
        if use_se:
            use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        
        self.depth = sum(layers[:]) * 3
        self.ts = [1] * layers[0] + [6] * sum(layers[1:])
        self.stem_channel = 32 / width_multiplier if width_multiplier < 1.0 else 32
        inplanes = input_filters / width_multiplier if width_multiplier < 1.0 else input_filters
        self.use_ses = use_ses
        self.width_multiplier = width_multiplier
        self.depth_multiplier = depth_multiplier
        self.se_ratio = se_ratio

        self.in_filters = []
        self.out_filters = []
        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                self.in_filters.append(int(round(self.stem_channel * width_multiplier)))
                self.out_filters.append(int(round(inplanes * width_multiplier)))
            else:
                self.in_filters.append(int(round(inplanes * width_multiplier)))
                inplanes += final_filters / (self.depth // 3 * 1.0)
                self.out_filters.append(int(round(inplanes * width_multiplier)))
    
    def build_model(self):
        def _norm(inp):
            rgb_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255], inp.dtype, [1, 1, 1, 3])
            rgb_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255], inp.dtype, [1, 1, 1, 3])
            return (inp - rgb_mean) * (1. / rgb_std)

        x = tf.keras.layers.Lambda(function=_norm, name="norm_input")(self.img_input)
        x = ConvNormActBlock(
            filters=int(round(self.stem_channel * self.width_multiplier)),
            kernel_size=3,
            strides=2,
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            activation=dict(activation="swish"),
            name="stem")(x)
        
        for idx, (in_f, out_f, t, s, se) in enumerate(zip(self.in_filters, self.out_filters, self.ts, self.strides, self.use_ses)):
            x = linear_bottleneck_v1(
                inputs=x,
                t=t,
                in_filters=in_f,
                out_filters=out_f,
                strides=s,
                use_se=se,
                se_ratio=self.se_ratio,
                kernel_initializer=self.kernel_initializer,
                data_format=self.data_format,
                normalization=self.normalization,
                name=str(idx))

        x = ConvNormActBlock(
            filters=int(round(1280 * self.width_multiplier)),
            kernel_size=1,
            strides=1,
            data_format=self.data_format,
            kernel_initializer=self.kernel_initializer,
            activation=dict(activation="swish"),
            name=str(len(self.in_filters)))(x)
        
        if self.data_format == "channels_last":
            x = tf.keras.layers.Lambda(
                function=lambda inp: tf.reduce_mean(inp, [1, 2], keepdims=True),
                name="global_avgpool")(x)
        else:
            x = tf.keras.layers.Lambda(
                function=lambda inp: tf.reduce_mean(inp, [2, 3], keepdims=True),
                name="global_avgpool")(x)
        x = tf.keras.layers.Conv2D(
            filters=self.num_classes, 
            kernel_size=1,
            kernel_initializer=self.kernel_initializer,
            name="logits")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=x, name=self.name)
        

@MODELS.register("ReXNet100")
def ReXNet100(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              strides=(2, 2, 2, 2, 2),
              num_classes=512,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    return ReXNet(
        name="rexnet_v1_1.0",
        use_se=True,
        se_ratio=12,
        input_filters=16,
        final_filters=180,
        width_multiplier=1.0,
        depth_multiplier=1.0,
        normalization=normalization,
        activation=activation,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()


@MODELS.register("ReXNet100")
def ReXNet130(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              strides=(2, 2, 2, 2, 2),
              num_classes=512,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    return ReXNet(
        name="rexnet_v1_1.3",
        use_se=True,
        se_ratio=12,
        input_filters=16,
        final_filters=180,
        width_multiplier=1.3,
        depth_multiplier=1.0,
        normalization=normalization,
        activation=activation,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()


@MODELS.register("ReXNet100")
def ReXNet150(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              strides=(2, 2, 2, 2, 2),
              num_classes=512,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    return ReXNet(
        name="rexnet_v1_1.5",
        use_se=True,
        se_ratio=12,
        input_filters=16,
        final_filters=180,
        width_multiplier=1.5,
        depth_multiplier=1.0,
        normalization=normalization,
        activation=activation,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()


@MODELS.register("ReXNet100")
def ReXNet200(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="relu"),
              strides=(2, 2, 2, 2, 2),
              num_classes=512,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              **kwargs):
    return ReXNet(
        name="rexnet_v1_2.0",
        use_se=True,
        se_ratio=12,
        input_filters=16,
        final_filters=180,
        width_multiplier=1.5,
        depth_multiplier=1.0,
        normalization=normalization,
        activation=activation,
        drop_rate=drop_rate,
        input_shape=input_shape,
        input_tensor=input_tensor,
        **kwargs).build_model()


def _get_weight_name_map(use_ses, ts):
    name_map = {
        "stem/conv2d/kernel:0":               "features.0.weight",
        "stem/batch_norm/gamma:0":            "features.1.weight",
        "stem/batch_norm/beta:0":             "features.1.bias",
        "stem/batch_norm/moving_mean:0":      "features.1.running_mean",
        "stem/batch_norm/moving_variance:0":  "features.1.running_var",
    }

    for idx in range(len(use_ses)): 
        i = 0
        if ts[idx] != 1:
            name_map["%d/conv1/conv2d/kernel:0" % idx]              =  "features.%d.out.%d.weight" % (idx + 3, i)
            i += 1
            name_map["%d/conv1/batch_norm/gamma:0" % idx]           =  "features.%d.out.%d.weight" % (idx + 3, i)
            name_map["%d/conv1/batch_norm/beta:0" % idx]            =  "features.%d.out.%d.bias" % (idx + 3, i)
            name_map["%d/conv1/batch_norm/moving_mean:0" % idx]     =  "features.%d.out.%d.running_mean" % (idx + 3, i)
            name_map["%d/conv1/batch_norm/moving_variance:0" % idx] =  "features.%d.out.%d.running_var" % (idx + 3, i)
            i += 2
        name_map["%d/conv2/conv2d/depthwise_kernel:0" % idx]    =  "features.%d.out.%d.weight" % (idx + 3, i)
        i += 1
        name_map["%d/conv2/batch_norm/gamma:0" % idx]           =  "features.%d.out.%d.weight" % (idx + 3, i)
        name_map["%d/conv2/batch_norm/beta:0" % idx]            =  "features.%d.out.%d.bias" % (idx + 3, i)
        name_map["%d/conv2/batch_norm/moving_mean:0" % idx]     =  "features.%d.out.%d.running_mean" % (idx + 3, i)
        name_map["%d/conv2/batch_norm/moving_variance:0" % idx] =  "features.%d.out.%d.running_var" % (idx + 3, i)
        i += 1
        if use_ses[idx]:
            name_map["%d/se/squeeze/kernel:0" % idx]               =  "features.%d.out.%d.fc.0.weight" % (idx + 3, i)
            name_map["%d/se/squeeze/bias:0" % idx]                 =  "features.%d.out.%d.fc.1.bias" % (idx + 3, i)
            name_map["%d/se/squeeze/norm/gamma:0" % idx]           =  "features.%d.out.%d.fc.1.weight" % (idx + 3, i)
            name_map["%d/se/squeeze/norm/beta:0" % idx]            =  "features.%d.out.%d.fc.1.bias" % (idx + 3, i)
            name_map["%d/se/squeeze/norm/moving_mean:0" % idx]     =  "features.%d.out.%d.fc.1.running_mean" % (idx + 3, i)
            name_map["%d/se/squeeze/norm/moving_variance:0" % idx] =  "features.%d.out.%d.fc.1.running_var" % (idx + 3, i)
            name_map["%d/se/excitation/kernel:0" % idx]            =  "features.%d.out.%d.fc.3.weight" % (idx + 3, i)
            name_map["%d/se/excitation/bias:0" % idx]              =  "features.%d.out.%d.fc.3.bias" % (idx + 3, i)
            i += 1
        i += 1
        name_map["%d/conv3/conv2d/kernel:0" % idx]              =  "features.%d.out.%d.weight" % (idx + 3, i)
        i += 1
        name_map["%d/conv3/batch_norm/gamma:0" % idx]           =  "features.%d.out.%d.weight" % (idx + 3, i)
        name_map["%d/conv3/batch_norm/beta:0" % idx]            =  "features.%d.out.%d.bias" % (idx + 3, i)
        name_map["%d/conv3/batch_norm/moving_mean:0" % idx]     =  "features.%d.out.%d.running_mean" % (idx + 3, i)
        name_map["%d/conv3/batch_norm/moving_variance:0" % idx] =  "features.%d.out.%d.running_var" % (idx + 3, i)
        idx += 1

    name_map["%d/conv2d/kernel:0" % idx]              =  "features.%d.weight" % (idx + 3)
    name_map["%d/batch_norm/gamma:0" % idx]           =  "features.%d.weight" % (idx + 4)
    name_map["%d/batch_norm/beta:0" % idx]            =  "features.%d.bias" % (idx + 4)
    name_map["%d/batch_norm/moving_mean:0" % idx]     =  "features.%d.running_mean" % (idx + 4)
    name_map["%d/batch_norm/moving_variance:0" % idx] =  "features.%d.running_var" % (idx + 4)
    
    name_map["logits/kernel:0"]  = "output.1.weight"
    name_map["logits/bias:0"]    = "output.1.bias"

    return name_map


def _get_weights_from_pretrained(model, pretrained_weights_path, use_ses, ts):
    import torch
    import numpy as np

    pretrained = torch.load(pretrained_weights_path, map_location="cpu")
    for k, v in pretrained.items():
        if "tracked" not in k:
            print(k, v.numpy().shape)
    name_map = _get_weight_name_map(use_ses, ts)
    # [print(k, v) for k, v in name_map.items()]
    
    for w in model.weights:
        name = w.name
        print(name, w.shape.as_list())
        # print(name_map[name], w.shape.as_list())
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
    name = "rexnetv1_1.5"
    
    rexnet = ReXNet(
        name=name,
        use_se=True,
        se_ratio=12,
        input_filters=16,
        final_filters=180,
        width_multiplier=1.5,
        depth_multiplier=1.0,
        input_shape=(224, 224, 3),
        num_classes=1000)

    model = rexnet.build_model()
    # model.summary()
    _get_weights_from_pretrained(
        model, 
        "/Users/bailang/Downloads/%s.pth" % name, 
        use_ses=rexnet.use_ses,
        ts=rexnet.ts)
    
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
