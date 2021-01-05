import math
import tensorflow  as tf 
from .model import Model
from .builder import MODELS
from .common import ConvNormActBlock
from core.layers import build_activation


class Bottle2neckX(tf.keras.Model):
    expansion = 4

    def __init__(self,
                 filters,
                 cardinality,
                 strides=1,
                 scale=4,
                 base_width=26,
                 dilation_rate=1,
                 data_format="channels_last",
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 downsample=False,
                 trainable=True,
                 dropblock=None,
                 stype="normal",
                 name="Bottle2neck"):
        super(Bottle2neckX, self).__init__(name=name)
        self.stype = stype
        self.scale = scale

        self.strides = strides

        width = int(math.floor(filters * (base_width / 64.)))
        self.channel_axis = -1 if data_format == "channels_last" else 1

        self.conv1 = ConvNormActBlock(filters=width * cardinality * scale,
                                      kernel_size=1,
                                      trainable=trainable,
                                      data_format=data_format,
                                      normalization=normalization,
                                      activation=activation,
                                      dropblock=dropblock,
                                      name="conv1")
       
        num_convs = scale if scale == 1 else scale - 1
        self.num_convs = num_convs
        self.convs = []
        for i in range(num_convs):
            conv = ConvNormActBlock(filters=width * cardinality, 
                                    kernel_size=3, 
                                    strides=strides, 
                                    data_format=data_format, 
                                    dilation_rate=dilation_rate if strides == 1 else 1, 
                                    trainable=trainable,
                                    groups=cardinality,
                                    normalization=normalization,
                                    activation=activation,
                                    dropblock=dropblock,
                                    name="convs/%d" % i)
            self.convs.append(conv)
        if stype == "stage":
            padding = "same"
            if strides != 1:
                self.pad = tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))
                padding = "valid"
            self.avgpool = tf.keras.layers.AvgPool2D(3, strides, padding, data_format, name=name + "/avgpool")
           
        self.conv3 = ConvNormActBlock(filters=filters * self.expansion,
                                      kernel_size=1,
                                      trainable=trainable,
                                      data_format=data_format,
                                      normalization=normalization,
                                      activation=None,
                                      dropblock=dropblock,
                                      name="conv3")
        self.act = build_activation(**activation, name=activation["activation"])
        if downsample:
            self.downsample = ConvNormActBlock(filters=filters * self.expansion,
                                               kernel_size=1,
                                               strides=strides,
                                               trainable=trainable,
                                               data_format=data_format,
                                               normalization=normalization,
                                               activation=None,
                                               dropblock=dropblock,
                                               name="downsample")

    def call(self, inputs, training=None):
        shortcut = inputs

        x = self.conv1(inputs, training=training)
        spx = tf.split(x, self.scale, self.channel_axis)
        for i in range(self.num_convs):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp += spx[i]
            sp = self.convs[i](sp, training=training)
            if i == 0:
                x = sp
            else:
                x = tf.concat([x, sp], axis=self.channel_axis)
        if self.scale != 1 and self.stype == "normal":
            x = tf.concat([x, spx[self.num_convs]], self.channel_axis)
        elif self.scale != 1 and self.stype == "stage":
            if hasattr(self, "pad"):
                x = tf.concat([x, self.avgpool(self.pad(spx[self.num_convs]))], self.channel_axis)
            else:
                x = tf.concat([x, self.avgpool(spx[self.num_convs])], self.channel_axis)
        x = self.conv3(x, training=training)

        if hasattr(self, "downsample"):
            shortcut = self.downsample(shortcut, training)
        
        x += shortcut
        x = self.act(x)

        return x



class Res2NeXt(Model):
    def __init__(self, 
                 name, 
                 num_blocks,
                 convolution='conv2d', 
                 dropblock=dict(block_size=7, drop_rate=0.1),
                 normalization=dict(normalization="batch_norm", momentum=0.9, epsilon=1e-5, axis=-1, trainable=True),
                 activation=dict(activation="relu"),
                 output_indices=(3, 4), 
                 strides=(2, 2, 2, 2, 2), 
                 dilation_rates=(1, 1, 1, 1, 1), 
                 frozen_stages=(-1, ), 
                 input_shape=None, 
                 input_tensor=None, 
                 cardinality=8,
                 base_width=26,
                 scale=4,
                 num_classes=1000, 
                 drop_rate=0.5):
        super(Res2NeXt, self).__init__(name, 
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
                                       drop_rate=drop_rate)
        self.num_blocks = num_blocks
        self.base_width = base_width
        self.scale = scale
        self.cardinality = cardinality

    def build_model(self):
        def _norm(inp):
            inp -= (tf.convert_to_tensor(self._rgb_mean * 255., inp.dtype))
            inp /= (tf.convert_to_tensor(self._rgb_std * 255., inp.dtype))
         
            return inp

        x = tf.keras.layers.Lambda(_norm, name="norm_input")(self.img_input)
        x = ConvNormActBlock(filters=64,
                             kernel_size=(7, 7),
                             strides=self.strides[0],
                             dilation_rate=self.dilation_rates[0],
                             trainable=1 not in self.frozen_stages,
                             kernel_initializer="he_normal",
                             normalization=self.normalization,
                             name="conv1")(x)
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", name="pool1")(x)
        self.infilters = 64 

        outputs = [x]
        x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = tf.keras.layers.MaxPool2D((3, 3), self.strides[1], "valid", self.data_format, name="maxpool")(x)
        x = self._make_layer(x, 64,  self.num_blocks[0], 1,               self.dilation_rates[1], 2 not in self.frozen_stages, name="layer1")
        outputs.append(x)
        x = self._make_layer(x, 128, self.num_blocks[1], self.strides[2], self.dilation_rates[2], 3 not in self.frozen_stages, name="layer2")
        outputs.append(x)
        x = self._make_layer(x, 256, self.num_blocks[2], self.strides[3], self.dilation_rates[3], 4 not in self.frozen_stages, name="layer3")
        outputs.append(x)
        x = self._make_layer(x, 512, self.num_blocks[3], self.strides[4], self.dilation_rates[4], 5 not in self.frozen_stages, name="layer4")
        outputs.append(x)

        if -1 not in self.output_indices:
            outputs = (outputs[i-1] for i in self.output_indices)
        else:
            x = tf.keras.layers.GlobalAvgPool2D(data_format=self.data_format, name="gloabl_avgpool")(x)
            if self.drop_rate and self.drop_rate > 0.:
                x = tf.keras.layers.Dropout(rate=self.drop_rate, name="drop")(x)
            outputs = tf.keras.layers.Dense(units=self.num_classes, name="logits")(x)

        return tf.keras.Model(inputs=self.img_input, outputs=outputs, name=self.name)
    
    def _make_layer(self, inputs, filters, num_block, strides=1, dilation_rate=1, trainable=True, name="layer"):
        x = Bottle2neckX(filters,
                         strides=strides,
                         base_width=self.base_width,
                         scale=self.scale,
                         dilation_rate=dilation_rate,
                         data_format=self.data_format,
                         trainable=trainable,
                         dropblock=self.dropblock,
                         normalization=self.normalization,
                         activation=self.activation,
                         downsample=strides != 1 or self.infilters != filters * Bottle2neckX.expansion,
                         stype="stage",
                         name=name + "/0")(inputs)
        
        for i in range(1, num_block):
            x = Bottle2neckX(filters,
                             strides=1,
                             base_width=self.base_width,
                             scale=self.scale,
                             dilation_rate=dilation_rate,
                             data_format=self.data_format,
                             trainable=trainable,
                             dropblock=self.dropblock,
                             normalization=self.normalization,
                             activation=self.activation,
                             downsample=False,
                             name=name + "/%d" % i)(x)
        self.infilters = filters * Bottle2neckX.expansion

        return x
    

@MODELS.register("Res2NeXt50")
def Res2NeXt50(convolution='conv2d', 
               dropblock=dict(block_size=7, drop_rate=0.1), 
               normalization=dict(normalization='batch_norm', momentum=0.9, epsilon=1e-05, axis = -1, trainable =True), 
               activation=dict(activation='relu'), 
               output_indices=(3, 4), strides=(2, 2, 2, 2, 2), 
               dilation_rates=(1, 1, 1, 1, 1), 
               frozen_stages=(-1, ), 
               input_shape=None, 
               input_tensor=None,
               num_classes=1000, 
               drop_rate=0.5):
    return Res2NeXt("res2next50", 
                    num_blocks=[3, 4, 6, 3],
                    convolution=convolution, 
                    dropblock=dropblock, 
                    normalization=normalization, 
                    activation=activation, 
                    output_indices=output_indices, 
                    strides=strides, 
                    dilation_rates=dilation_rates, 
                    frozen_stages=frozen_stages, 
                    input_shape=input_shape, 
                    input_tensor=input_tensor, 
                    base_width=4, 
                    cardinality=8,
                    scale=4, 
                    num_classes=num_classes, 
                    drop_rate=drop_rate).build_model()


def _get_weight_name_map(blocks, scale):
    name_map = {
        "conv1/conv2d/kernel:0": "conv1.weight",
        "conv1/batch_norm/gamma:0": "bn1.weight",
        "conv1/batch_norm/beta:0": "bn1.bias",
        "conv1/batch_norm/moving_mean:0": "bn1.running_mean",
        "conv1/batch_norm/moving_variance:0": "bn1.running_var"
    }

    for i in range(1, 5):
        for j in range(blocks[i - 1]):
            for k in range(1, 4):
                n1 = "layer%d/%d/conv%d" % (i, j, k)
                n2 = "layer%d.%d" % (i, j)
                if k != 2:
                    m = {
                        "%s/conv2d/kernel:0" % n1: "%s.conv%d.weight" % (n2, k), 
                        "%s/batch_norm/gamma:0" % n1: "%s.bn%d.weight" % (n2, k),
                        "%s/batch_norm/beta:0" % n1: "%s.bn%d.bias" % (n2, k),
                        "%s/batch_norm/moving_mean:0" % n1: "%s.bn%d.running_mean" % (n2, k),
                        "%s/batch_norm/moving_variance:0" % n1: "%s.bn%d.running_var" % (n2, k),
                        "layer%d/0/downsample/conv2d/kernel:0" % i: "layer%d.0.downsample.0.weight" % i,
                        "layer%d/0/downsample/batch_norm/gamma:0" % i: "layer%d.0.downsample.1.weight" % i,
                        "layer%d/0/downsample/batch_norm/beta:0" % i: "layer%d.0.downsample.1.bias" % i,
                        "layer%d/0/downsample/batch_norm/moving_mean:0" % i: "layer%d.0.downsample.1.running_mean" % i,
                        "layer%d/0/downsample/batch_norm/moving_variance:0" % i: "layer%d.0.downsample.1.running_var" % i
                    }
                    name_map.update(m)
                else:
                    for s in range(scale - 1):
                        m = {
                            "layer%d/%d/convs/%d/conv2d/kernel:0" % (i, j, s): "%s.convs.%d.weight" % (n2, s), 
                            "layer%d/%d/convs/%d/batch_norm/gamma:0" % (i, j, s): "%s.bns.%d.weight" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/beta:0" % (i, j, s): "%s.bns.%d.bias" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/moving_mean:0" % (i, j, s): "%s.bns.%d.running_mean" % (n2, s),
                            "layer%d/%d/convs/%d/batch_norm/moving_variance:0" % (i, j, s): "%s.bns.%d.running_var" % (n2, s),
                        }
                        name_map.update(m)
    
    name_map["logits/kernel:0"] = "fc.weight"
    name_map["logits/bias:0"] = "fc.bias"

    return name_map


def _torch2h5(model, torch_weight_path, blocks, scale):
    import torch
    import numpy as np

    net = torch.load(torch_weight_path, map_location=torch.device('cpu'))
    
    # for k, _ in net.items():
    #     if "tracked" in k:
    #         continue
    #     print(k) 

    name_map = _get_weight_name_map(blocks, scale)
    for weight in model.weights:
        name = weight.name
        
        tw = net[name_map[name]].numpy()
        if len(tw.shape) == 4:
            tw = np.transpose(tw, (2, 3, 1, 0))
        if len(tw.shape) == 2:
            tw = np.transpose(tw, (1, 0))
        
        weight.assign(tw)

    del net


if __name__ == "__main__":
    name = "res2net50_48w_2s"
    blocks = [3, 4, 6, 3]
    scale = 2
    model = Res2NeXt50(input_shape=(224, 224, 3), output_indices=(-1, ))
    # model(tf.random.uniform([1, 224, 224, 3], 0, 255))
    model.summary()
    _torch2h5(model, "/Users/bailang/Downloads/pretrained_weights/%s.pth" % name, blocks, scale)

    with tf.io.gfile.GFile("/Users/bailang/Documents/pandas.jpg", "rb") as gf:
        images = tf.image.decode_jpeg(gf.read())

    images = tf.image.resize(images, (224, 224))[None]
    logits = model(images, training=False)
    probs = tf.nn.softmax(logits)
    print(tf.nn.top_k(tf.squeeze(probs), k=5))