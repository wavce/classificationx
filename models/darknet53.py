import os
import numpy as np
import tensorflow as tf
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from core.layers import build_activation 


def residual_block(inputs, 
                   filters, 
                   dilation_rate=1,
                   expension=2,
                   data_format="channels_last",
                   normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-3, axis=-1, trainable=True),
                   activation=dict(activation="leaky_relu", alpha=0.1),
                   kernel_initializer="glorot_uniform",
                   trainable=True,
                   index=0):
    x = ConvNormActBlock(filters=filters // expension,
                         kernel_size=1,
                         dilation_rate=1,
                         normalization=normalization,
                         activation=activation,
                         trainable=trainable,
                         kernel_initializer="glorot_uniform",
                         name="conv" + str(index))(inputs)
    x = ConvNormActBlock(filters=filters,
                         kernel_size=3,
                         dilation_rate=dilation_rate,
                         normalization=normalization,
                         activation=activation,
                         trainable=trainable,
                         kernel_initializer="glorot_uniform",
                         name="conv" + str(index + 1))(x)
    x = tf.keras.layers.Add(name="add" + str((index + 1) // 2))([inputs, x])

    return x


@MODELS.register("DarkNet53")
def darknet53(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
              activation=dict(activation="leaky_relu", alpha=0.1),
              output_indices=(-1, ),
              strides=(2, 2, 2, 2, 2),
              dilation_rates=(1, 1, 1, 1, 1),
              frozen_stages=(-1,),
              dropblock=None,
              num_classes=1000,
              drop_rate=0.5,
              input_shape=(224, 224, 3),
              input_tensor=None,
              data_format="channels_last"):
    
    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor
    

    def _norm(inp):
        inp /= 255.
        return inp

    x = tf.keras.layers.Lambda(_norm, name="norm_input")(inputs)
    x = ConvNormActBlock(filters=32,
                         kernel_size=3,
                         strides=1,
                         dilation_rate=dilation_rates[0],
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv1")(x)
    x = ConvNormActBlock(filters=64,
                         kernel_size=3,
                         strides=strides[0],
                         dilation_rate=1,
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv2")(x)
    x = residual_block(x, 64, dilation_rates[0], 2, data_format, normalization, activation, "glorot_uniform", 1 not in frozen_stages, 3)
    outputs = [x]

    x = ConvNormActBlock(filters=128,
                         kernel_size=3,
                         strides=strides[1],
                         dilation_rate=1,
                         trainable=2 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv5")(x)
    for i in range(2):
        x = residual_block(x, 128, dilation_rates[1], 2, data_format, normalization, activation, "glorot_uniform", 2 not in frozen_stages, 6 + i * 2)
    outputs.append(x)

    x = ConvNormActBlock(filters=256,
                         kernel_size=3,
                         strides=strides[2],
                         dilation_rate=1,
                         trainable=3 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv10")(x)
    for i in range(8):
        x = residual_block(x, 256, dilation_rates[2], 2, data_format, normalization, activation, "glorot_uniform", 3 not in frozen_stages, 11 + i * 2)
    outputs.append(x)

    x = ConvNormActBlock(filters=512,
                         kernel_size=3,
                         strides=strides[3],
                         dilation_rate=1,
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv27")(x)
    for i in range(8):
        x = residual_block(x, 512, dilation_rates[3], 2, data_format, normalization, activation, "glorot_uniform", 4 not in frozen_stages, 28 + i * 2)
    outputs.append(x)

    x = ConvNormActBlock(filters=1024,
                         kernel_size=3,
                         strides=strides[4],
                         dilation_rate=1,
                         trainable=5 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv44")(x)
    for i in range(4):
        x = residual_block(x, 1024, dilation_rates[4], 2, data_format, normalization, activation, "glorot_uniform", 5 not in frozen_stages, 45 + i * 2)
    outputs.append(x)

    if -1 in output_indices:
        pool_axis = [1, 2] if data_format == "channels_last" else [2, 3]
        x = tf.keras.layers.Lambda(lambda inp: tf.reduce_mean(inp, pool_axis, keepdims=True), name="global_avgpool")(x)
        x = tf.keras.layers.Dropout(rate=drop_rate)(x)
        outputs = tf.keras.layers.Conv2D(num_classes, 1, 1, data_format=data_format, name="conv53")(x)
    else:
        outputs = (outputs[i-1] for i in output_indices)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


@MODELS.register("CSPDarkNet53")
def csp_darknet53(normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                  activation=dict(activation="leaky_relu", alpha=0.1),
                  output_indices=(-1, ),
                  strides=(2, 2, 2, 2, 2),
                  dilation_rates=(1, 1, 1, 1, 1),
                  frozen_stages=(-1,),
                  dropblock=None,
                  num_classes=1000,
                  drop_rate=0.5,
                  input_shape=(224, 224, 3),
                  input_tensor=None,
                  data_format="channels_last"):
    
    if input_tensor is None:
        inputs = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            inputs = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            inputs = input_tensor
    
    channels_axis = -1 if data_format == "channels_last" else 1

    x = tf.keras.layers.Lambda(lambda inp: inp / 255., name="norm_input")(inputs)
    x = ConvNormActBlock(filters=32,
                         kernel_size=3,
                         strides=1,
                         dilation_rate=dilation_rates[0],
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv1")(x)
    x = ConvNormActBlock(filters=64,
                         kernel_size=3,
                         strides=strides[0],
                         dilation_rate=1,
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv2")(x)
    route = ConvNormActBlock(filters=64,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rates[0],
                             trainable=1 not in frozen_stages,
                             kernel_initializer="glorot_uniform",
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv3")(x)
    x = ConvNormActBlock(filters=64,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[0],
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv4")(x)
    x = residual_block(x, 64, dilation_rates[0], 1, data_format, normalization, activation, "glorot_uniform", 1 not in frozen_stages, 5)
    x = ConvNormActBlock(filters=64,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[0],
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv7")(x)
    x = tf.keras.layers.Concatenate(axis=channels_axis)([x, route])
    x = ConvNormActBlock(filters=64,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[0],
                         trainable=1 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv8")(x)
    outputs = [x]

    # Downsample
    x = ConvNormActBlock(filters=128,
                         kernel_size=3,
                         strides=strides[1],
                         dilation_rate=1,
                         trainable=2 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv9")(x)
    route = ConvNormActBlock(filters=64,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rates[1],
                             trainable=2 not in frozen_stages,
                             kernel_initializer="glorot_uniform",
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv10")(x)
    x = ConvNormActBlock(filters=64,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[1],
                         trainable=2 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv11")(x)
    for i in range(2):
        x = residual_block(x, 64, dilation_rates[1], 1, data_format, normalization, activation, "glorot_uniform", 2 not in frozen_stages, 12 + i * 2)
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[1],
                         trainable=2 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv16")(x)
    x = tf.keras.layers.Concatenate(axis=channels_axis)([x, route])
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[1],
                         trainable=2 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv17")(x)
    outputs.append(x)

    x = ConvNormActBlock(filters=256,
                         kernel_size=3,
                         strides=strides[2],
                         dilation_rate=1,
                         trainable=3 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv18")(x)
    route = ConvNormActBlock(filters=128,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rates[2],
                             trainable=3 not in frozen_stages,
                             kernel_initializer="glorot_uniform",
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv19")(x)
    x = ConvNormActBlock(filters=128,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[2],
                         trainable=3 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv20")(x)
    for i in range(8):
        x = residual_block(x, 256, dilation_rates[2], 1, data_format, normalization, activation, "glorot_uniform", 3 not in frozen_stages, 21 + i * 2)
    x = ConvNormActBlock(filters=256,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[2],
                         trainable=3 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv37")(x)
    x = tf.keras.layers.Concatenate(axis=channels_axis)([x, route])
    x = ConvNormActBlock(filters=256,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[2],
                         trainable=3 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv38")(x)
    outputs.append(x)

    x = ConvNormActBlock(filters=512,
                         kernel_size=3,
                         strides=strides[3],
                         dilation_rate=1,
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv39")(x)
    route = ConvNormActBlock(filters=256,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rates[3],
                             trainable=4 not in frozen_stages,
                             kernel_initializer="glorot_uniform",
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv40")(x)
    x = ConvNormActBlock(filters=256,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[3],
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv41")(x)
    for i in range(8):
        x = residual_block(x, 256, dilation_rates[3], 1, data_format, normalization, activation, "glorot_uniform", 4 not in frozen_stages, 42 + i * 2)
    x = ConvNormActBlock(filters=256,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[3],
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv58")(x)
    x = tf.keras.layers.Concatenate(axis=channels_axis)([x, route])
    x = ConvNormActBlock(filters=512,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[3],
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv59")(x)
    outputs.append(x)

    x = ConvNormActBlock(filters=1024,
                         kernel_size=3,
                         strides=strides[4],
                         dilation_rate=1,
                         trainable=5 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv60")(x)
    route = ConvNormActBlock(filters=512,
                             kernel_size=1,
                             strides=1,
                             dilation_rate=dilation_rates[4],
                             trainable=5 not in frozen_stages,
                             kernel_initializer="glorot_uniform",
                             normalization=normalization,
                             activation=activation,
                             data_format=data_format,
                             name="conv61")(x)
    x = ConvNormActBlock(filters=512,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[4],
                         trainable=5 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv62")(x)
    for i in range(4):
        x = residual_block(x, 512, dilation_rates[4], 1, data_format, normalization, activation, "glorot_uniform", 5 not in frozen_stages, 63 + i * 2)
    x = ConvNormActBlock(filters=512,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[3],
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv71")(x)
    x = tf.keras.layers.Concatenate(axis=channels_axis)([x, route])
    x = ConvNormActBlock(filters=1024,
                         kernel_size=1,
                         strides=1,
                         dilation_rate=dilation_rates[3],
                         trainable=4 not in frozen_stages,
                         kernel_initializer="glorot_uniform",
                         normalization=normalization,
                         activation=activation,
                         data_format=data_format,
                         name="conv72")(x)
    outputs.append(x)

    if -1 in output_indices:
        pool_axis = [1, 2] if data_format == "channels_last" else [2, 3]
        x = tf.keras.layers.Lambda(lambda inp: tf.reduce_mean(inp, pool_axis, keepdims=True), name="global_avgpool")(x)
        x = tf.keras.layers.Dropout(rate=drop_rate)(x)
        outputs = tf.keras.layers.Conv2D(num_classes, 1, 1, data_format=data_format, name="conv73")(x)
    else:
        outputs = (outputs[i-1] for i in output_indices)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _load_darknet_weights(model, darknet_weights_path, num_convs):

    wf = open(darknet_weights_path, "rb")
    major, minor, revision, seen, _ = np.fromfile(wf, np.int32, 5)

    for i in range(num_convs):
        layer = model.get_layer("conv" + str(i+1))
        if isinstance(layer, ConvNormActBlock):
            kernel = layer.conv.kernel
            gamma = layer.norm.gamma
            beta = layer.norm.beta
            moving_mean = layer.norm.moving_mean
            moving_variance = layer.norm.moving_variance
            
            ksize, _, infilters, filters = kernel.shape.as_list()
            dshape = (filters, infilters, ksize, ksize)

            beta.assign(np.fromfile(wf, np.float32, filters))
            gamma.assign(np.fromfile(wf, np.float32, filters))
            moving_mean.assign(np.fromfile(wf, np.float32, filters))
            moving_variance.assign(np.fromfile(wf, np.float32, filters))

            dkernel = np.fromfile(wf, np.float32, np.product(dshape))
            dkernel = dkernel.reshape(dshape).transpose([2, 3, 1, 0])
            kernel.assign(dkernel)
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            kernel = layer.kernel
            bias = layer.bias

            ksize, _, infilters, filters = kernel.shape.as_list()
            dshape = (filters, infilters, ksize, ksize)
            bias.assign(np.fromfile(wf, np.float32, filters))

            dkernel = np.fromfile(wf, np.float32, np.product(dshape))
            dkernel = dkernel.reshape(dshape).transpose([2, 3, 1, 0])
            kernel.assign(dkernel)

    assert len(wf.read()) == 0, "Failed to read all data"
    wf.close()


if __name__ == '__main__':
    from .common import fuse

    model = csp_darknet53(input_shape=(256, 256, 3),
                          normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                          activation=dict(activation="leaky_relu", alpha=0.1))
    
    model.summary()
    _load_darknet_weights(model, "/Users/bailang/Downloads/pretrained_weights/csdarknet53.weights", 53)
    
    # fuse(model, block_fn)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (256, 256))
    images = tf.expand_dims(images, axis=0)
    lbl = model(images, training=False)
    top5prob, top5class = tf.nn.top_k(tf.squeeze(tf.nn.softmax(lbl, -1), axis=0), k=5)
    print("prob:", top5prob.numpy())
    print("class:", top5class.numpy())
    
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/darknet53.h5")
    model.save_weights("/Users/bailang/Downloads/pretrained_weights/darknet53/model.ckpt")

