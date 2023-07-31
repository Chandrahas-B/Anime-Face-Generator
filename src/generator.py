from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def transpose(x, f, ks, s, alpha= 0.3, ki = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)):
    x = tf.keras.layers.Conv2DTranspose(f, ks, strides= s, padding= 'same', kernel_initializer = ki)(x)
#     x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    return x

def conv(x, f, ks, alpha= 0.7):
    x = tf.keras.layers.Conv2D(f, ks, padding= 'same')(x)
#     x = tf.keras.layers.LeakyReLU(alpha)(x)
    x = tf.keras.layers.ReLU()(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    return x


latent_dim = 300

def Generator(latent_dim = latent_dim):
    input_ = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 512)(input_)
    x = layers.ReLU()(x)
    x = layers.Reshape((8, 8, 512))(x)
    x = transpose(x, 256, 4, 2)
    x = conv(x, 512, 4)
    x = transpose(x, 128, 4, 2)
    x = conv(x, 256, 3)
    x = transpose(x, 64, 4, 2)
    x = conv(x, 16, 2)
    output = layers.Conv2D(3, (4, 4), padding='same', activation='tanh')(x)

    model = keras.Model(inputs=input_, outputs=output, name='generator')

    return model