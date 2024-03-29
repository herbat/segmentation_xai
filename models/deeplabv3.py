
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Lambda, Conv2D, Conv2DTranspose, Activation, Reshape, \
    concatenate, Concatenate, BatchNormalization, ZeroPadding2D
from keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2


base_models = {
    'resnet50': ResNet50V2,
    'resnet101': ResNet101V2,
    'resnet152': ResNet152V2,
}


def upsample(tensor, size):
    """bilinear upsampling"""
    name = tensor.name.split('/')[0] + '_upsample'

    def bilinear_upsample(x, _size):
        resized = tf.image.resize(
            images=x, size=_size)
        return resized

    y = Lambda(lambda x: bilinear_upsample(x, size),
               output_shape=size, name=name)(tensor)
    return y


def aspp(tensor):
    """atrous spatial pyramid pooling"""

    dims = K.int_shape(tensor)

    y_pool = AveragePooling2D(pool_size=(
        dims[1], dims[2]), name='average_pooling')(tensor)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)

    y_pool = upsample(tensor=y_pool, size=[dims[1], dims[2]])

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
    y_1 = BatchNormalization(name=f'bn_2')(y_1)
    y_1 = Activation('relu', name=f'relu_2')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                 kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
    y_6 = BatchNormalization(name=f'bn_3')(y_6)
    y_6 = Activation('relu', name=f'relu_3')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
    y_12 = BatchNormalization(name=f'bn_4')(y_12)
    y_12 = Activation('relu', name=f'relu_4')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                  kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
    y_18 = BatchNormalization(name=f'bn_5')(y_18)
    y_18 = Activation('relu', name=f'relu_5')(y_18)

    y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
               kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
    y = BatchNormalization(name=f'bn_final')(y)
    y = Activation('relu', name=f'relu_final')(y)
    return y


class DeepLabV3Plus(Model):
    def __init__(self, img_height, img_width, nclasses: int = 66, _base_model: str = 'resnet50'):
        self.img_width = img_width
        self.img_height = img_height
        inp, out = self.build_model(_base_model)
        super().__init__(inputs=inp, outputs=out)
        self.nclasses = nclasses

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def build_model(self, _base_model: str = 'resnet50'):
        print('*** Building DeepLabv3Plus Network ***')

        base_model = base_models[_base_model](input_shape=(self.img_height, self.img_width, 3),
                                              weights='imagenet',
                                              include_top=False)
        base_model.summary()
        layer_names = [layer['name'] for layer in base_model.get_config()['layers']]
        print(len(layer_names))
        image_features = base_model.get_layer('post_relu').output
        x_a = aspp(image_features)
        x_a = upsample(tensor=x_a, size=[self.img_height // 4, self.img_width // 4])

        x_b = base_model.get_layer('activation_9').output
        x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                     kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
        x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
        x_b = Activation('relu', name='low_level_activation')(x_b)

        x = concatenate([x_a, x_b], name='decoder_concat')

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
        x = BatchNormalization(name=f'bn_decoder_1')(x)
        x = Activation('relu', name='activation_decoder_1')(x)

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                   kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
        x = BatchNormalization(name=f'bn_decoder_2')(x)
        x = Activation('relu', name='activation_decoder_2')(x)
        x = upsample(x, [self.img_height, self.img_width])

        x = Conv2D(self.nclasses, (1, 1), name='output_layer')(x)
        '''
        x = Activation('softmax')(x) 
        tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        Args:
            from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
            we assume that `y_pred` encodes a probability distribution.
        '''
        return base_model.inputs, x

    def __call__(self, x: np.ndarray, *args, **kwargs):
        return self.model(x)

    def predict_gen(self, x: np.ndarray,):
        if x.ndim < 4:
            x = np.expand_dims(x, axis=0)

        return self.model.predict(x)

