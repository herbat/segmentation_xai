import numpy as np
from tensorflow import keras
from keras.utils import generic_utils

import segmentation_models as sm


class PSPNetModel(keras.Model):

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self, classes: int,
                 input_shape: tuple,
                 backbone: str = 'resnext101'):
        super().__init__()
        self.in_shape = input_shape
        self.model = sm.PSPNet(backbone, classes=classes, input_shape=input_shape, encoder_weights='imagenet')

    def __call__(self, x: np.ndarray, *args, **kwargs):
        return self.model(x)

    def predict_gen(self, x: np.ndarray,):
        if x.ndim < 4:
            x = np.expand_dims(x, axis=0)

        return self.model.predict(x)