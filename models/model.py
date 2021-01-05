import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self,
                 name,
                 convolution="conv2d",
                 normalization=dict(),
                 activation=dict(),
                 output_indices=(3, 4),
                 strides=(2, 2, 2, 2, 2),
                 dilation_rates=(1, 1, 1, 1, 1),
                 frozen_stages=(-1,),
                 input_shape=None,
                 input_tensor=None,
                 dropblock=None,
                 weight_decay=0.0001,
                 num_classes=1000,
                 drop_rate=0.5):
        """The model base class.

        Args:
            convolution: (str) the convolution using in backbone.
            normalization: (dict) the normalization layer, default None, if None, means not use normalization.
            activation: (dict) activation name.
            output_indices: (list[tuple]) the indices for outputs, e.g. [3, 4, 5] means
                output the stage 3, stage 4 and stage 5 in backbone.
            strides: (list[tuple]) the strides for every stage in backbone, e.g. [1, 1, 1, 1, 1].
            dilation_rates: (list[tuple]) the dilation_rates for every stage in backbone.
            frozen_stages: (list[tuple]) the indices for which stage should be frozen,
                e.g. [1, 2, 3] means frozen stage 1, stage 2 and stage 3.
            frozen_batch_normalization: (bool) Does frozen batch normalization.
        """
        assert isinstance(output_indices, (list, tuple))
        assert isinstance(strides, (list, tuple))
        assert isinstance(frozen_stages, (list, tuple))
        assert isinstance(dilation_rates, (list, tuple))

        self.name = name
        self.output_indices = output_indices
        self.strides = strides
        self.frozen_stages = frozen_stages
        self.dilation_rates = dilation_rates
        self.normalization = normalization 
        self.convolution = convolution
        self.activation = activation
        self.dropblock = dropblock
        self.weight_decay = weight_decay 
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self._rgb_mean = np.array([0.485, 0.456, 0.406])
        self._rgb_std = np.array([0.229, 0.224, 0.225])

        input_shape = (input_shape if isinstance(input_shape, (list, tuple)) 
                                        else [input_shape, input_shape, 3])
        if input_tensor is None:
            img_input = tf.keras.layers.Input(shape=input_shape)
        else:
            if not tf.keras.backend.is_keras_tensor(input_tensor):
                img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor
                
        self.img_input = img_input
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.data_format = tf.keras.backend.image_data_format()
    