import tensorflow as tf
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec


class GroupConv2D(tf.keras.layers.Conv2D):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=(1,1), 
                 padding='valid', 
                 data_format=None, 
                 dilation_rate=(1,1), 
                 activation=None, 
                 use_bias=True, 
                 group=1,
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 **kwargs):
        super().__init__(filters, 
                         kernel_size, 
                         strides=strides, 
                         padding=padding, 
                         data_format=data_format, 
                         dilation_rate=dilation_rate, 
                         activation=activation, 
                         use_bias=use_bias, 
                         kernel_initializer=kernel_initializer, 
                         bias_initializer=bias_initializer, 
                         kernel_regularizer=kernel_regularizer, 
                         bias_regularizer=bias_regularizer, 
                         activity_regularizer=activity_regularizer, 
                         kernel_constraint=kernel_constraint, 
                         bias_constraint=bias_constraint, 
                         **kwargs)
        
        self.group = group

    def build(self, input_shape):
        if self.group > 1: 
            input_shape = tensor_shape.TensorShape(input_shape)
            input_channel = self._get_input_channel(input_shape)
            kernel_shape = self.kernel_size + (input_channel // self.group, self.filters)

            self.kernel = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)
            if self.use_bias:
                self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True,
                    dtype=self.dtype)
            else:
                self.bias = None

            channel_axis = self._get_channel_axis()
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_channel})

            self._build_conv_op_input_shape = input_shape
            self._build_input_channel = input_channel
            self._padding_op = self._get_padding_op()
            self._conv_op_data_format = conv_utils.convert_data_format(
                self.data_format, self.rank + 2)

            if self.data_format == "channels_last":
                input_shapes = [
                    tensor_shape.TensorShape([input_shape[0], input_shape[1], input_shape[2], input_shape[3] // self.group])
                ] * self.group
            elif self.data_format == "channels_first":
                input_shapes = [
                    tensor_shape.TensorShape([input_shape[0] // self.group, input_shape[1], input_shape[2], input_shape[3]])
                ] * self.group
            else:
                raise ValueError('Invalid data_format:', self.data_format)
            
            self._channel_axis = channel_axis
            filter_shape = tensor_shape.TensorShape(self.kernel_size + (input_channel // self.group, self.filters // self.group))
            # print(self.kernels)
            self._convolution_ops = [
                nn_ops.Convolution(
                    input_shapes[i],
                    filter_shape=filter_shape,
                    dilation_rate=self.dilation_rate,
                    strides=self.strides,
                    padding=self._padding_op,
                    data_format=self._conv_op_data_format)
                for i in range(self.group)]
            self.built = True
        else:
            super(GroupConv2D, self).build(input_shape)
    
    def call(self, inputs):
        if self.group > 1:
            if self._recreate_conv_op(inputs):
                input_shape = inputs.get_shape()
                if self.data_format == "channels_last":
                    input_shapes = [
                        tensor_shape.TensorShape([input_shape[0], input_shape[1], input_shape[2], input_shape[3] // self.group])
                    ] * self.group
                elif self.data_format == "channels_first":
                    input_shapes = [
                        tensor_shape.TensorShape([input_shape[0] // self.group, input_shape[1], input_shape[2], input_shape[3]])
                    ] * self.group
                else:
                    raise ValueError('Invalid data_format:', self.data_format)

                self._convolution_ops = [
                    nn_ops.Convolution(
                        input_shapes[i],
                        filter_shape=self.kernels[i].shape,
                        dilation_rate=self.dilation_rate,
                        strides=self.strides,
                        padding=self._padding_op,
                        data_format=self._conv_op_data_format)
                for i in range(self.group)]

            kernels = tf.split(self.kernel, self.group, self._channel_axis)
            inputs = tf.split(inputs, self.group, self._channel_axis)
            outputs = [
                self._convolution_ops[i](inputs[i], kernels[i]) for i in range(self.group)
            ]
            outputs = tf.concat(outputs, self._channel_axis)

            if self.use_bias:
                if self.data_format == 'channels_first':
                    if self.rank == 1:
                        # nn.bias_add does not accept a 1D input tensor.
                        bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                        outputs += bias
                    else:
                        outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

            if self.activation is not None:
                return self.activation(outputs)
            return outputs
        else:
            return super(GroupConv2D, self).call(inputs)
