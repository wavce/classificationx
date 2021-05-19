import string
import numpy as np
import tensorflow as tf 
from .model import Model 
from .builder import MODELS
import torch.nn as nn
nn.MultiheadAttention


class PositionEmbedding(tf.keras.layers.Layer):
    """Applies AddPositionEmbs module.
       
       By default this layer uses a fixed sinusoidal embedding table. If a
       learned position embedding is desired, pass an initializer to  position_embedding_initializer.
       
       Args:
            positions: input position indices for packed sequences.
            position_embedding_initializer: positional embedding initializer.
        Returns:
            output: `(bs, timesteps, in_dim)`
    """
    def __init__(self, positions=None, position_embedding_initializer=None, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.positions = positions
        self.position_embedding_initializer = position_embedding_initializer
    
    def build(self, input_shape):
        embedding_shape = (1, input_shape[1], input_shape[2])
        self.pos_embeding = self.add_weight(name="pos_embedding", 
                                            shape=embedding_shape,
                                            dtype=self.dtype, 
                                            initializer=self.position_embedding_initializer)
        super(PositionEmbedding, self).build(input_shape)
    
    def call(self, inputs, training=None):
        if self.positions is None:
            return inputs + self.pos_embeding
        
        return inputs + tf.gather(self.pos_embeding, self.positions)
    
    def get_config(self):
        config = {
            "position_embedding_initializer": tf.keras.initializers.serialize(self.position_embedding_initializer),
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, token_dim, initializer=None, **kwargs):
        super(TokenEmbedding, self).__init__(**kwargs)

        self.token_dim = token_dim
        self.initializer = initializer
    
    def build(self, input_shape):
        self.token = self.add_weight(name="cls", shape=(1, 1, self.token_dim), initializer=self.initializer)
        super(TokenEmbedding, self).__init__(input_shape)
    
    def get_config(self):
        config = {
            "initializer": tf.keras.initializers.serialize(self.initializer)
        }
        base_config = super(TokenEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        bs = tf.shape(inputs)[0]
        token = tf.tile(self.token, [bs, 1, 1])
        
        return tf.concat([token, inputs], axis=1)


class MLP(tf.keras.Model):
    def __init__(self, in_dims, hidden_dims=None, out_dims=None, activation="gelu", dropout_rate=0.1, **kwargs):
        super(MLP, self).__init__(**kwargs)

        out_dims = out_dims or in_dims
        hidden_dims = hidden_dims or in_dims

        self.dense1 = tf.keras.layers.Dense(hidden_dims, activation, name="fc1")
        self.dense2 = tf.keras.layers.Dense(out_dims, name="fc2")
        self.dropout = tf.keras.layers.Dropout(dropout_rate, name="droupout")
    
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        
        return x


class Encoder1D(tf.keras.Model):
    def __init__(self, num_heads, key_dim, in_dim, mlp_dim, dropout_rate=0.1, attention_dropout_rate=0.1, **kwargs):
        super(Encoder1D, self).__init__(**kwargs)

        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads, key_dim, dropout=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(in_dim, mlp_dim, dropout_rate=dropout_rate)
    
    def call(self, inputs, training=None):
        x = self.layer_norm1(inputs)
        x = self.multi_head_attention(query=x, value=x, training=training)
        x = self.dropout(x, training=training)

        x += inputs

        y = self.layer_norm2(x)
        y = self.mlp(y, training=training)

        return x + y


class VisionTransformer(object):
    def __init__(self, 
                 input_shape, 
                 num_layers, 
                 num_heads, 
                 patch_size, 
                 hidden_size, 
                 mlp_dim,
                 num_classes=1000,
                 dropout_rate=0.1, 
                 attention_dropout_rate=0.1,
                 classifier="token"):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.mlp_dim = mlp_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.attention_dropout_rate = attention_dropout_rate
        self.classifier = classifier
        self.num_classes = num_classes

    def build_model(self):
        inputs = tf.keras.Input((self.input_shape[0], self.input_shape[1], 3))
        x = tf.keras.layers.Conv2D(filters=self.hidden_size,
                                   kernel_size=self.patch_size, 
                                   strides=self.patch_size,
                                   padding="valid",
                                   name="embedding")(inputs)
        
        _, h, w, c = tf.keras.backend.int_shape(x)
        x = tf.keras.layers.Reshape([h * w, c])(x)
        if self.classifier == "token":
            x = TokenEmbedding(token_dim=c)(x)
        
        x = PositionEmbedding(positions=None)(x)
        x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        _, _, c = tf.keras.backend.int_shape(x)
        for lyr in range(self.num_layers):
            x = Encoder1D(self.num_heads, 
                          in_dim=c, 
                          key_dim=self.hidden_size // self.num_heads,
                          mlp_dim=self.mlp_dim, 
                          dropout_rate=self.dropout_rate,
                          attention_dropout_rate=self.attention_dropout_rate,
                          name="encoder_{}".format(lyr))(x)
        x = tf.keras.layers.LayerNormalization(name="encoder_norm")(x)

        x = tf.keras.layers.Dense(units=self.num_classes, name="head", kernel_initializer="zeros")(x[:, 0])

        return tf.keras.Model(inputs=inputs, outputs=x)
        

def ViTB16(input_shape=(224, 224), num_classes=1000):
    return VisionTransformer(input_shape,
                             num_layers=12,
                             num_heads=12,
                             hidden_size=768,
                             patch_size=(16, 16),
                             mlp_dim=3072,
                             num_classes=num_classes,
                             dropout_rate=0.1,
                             attention_dropout_rate=0.,
                             classifier="token").build_model()


if __name__ == "__main__":
    model = ViTB16()
    model.summary()
