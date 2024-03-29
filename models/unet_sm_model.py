import numpy as np
from tensorflow import keras
from keras.utils import generic_utils

import segmentation_models as sm


def gen_spec(gen) -> tuple:
    while True:
        x, y, m = next(gen)
        yield np.repeat(x, 3, axis=3), y


class UnetModel(keras.Model):

    def get_config(self):
        pass

    def call(self, inputs, training=None, mask=None):
        pass

    def __init__(self, classes: int,
                 input_shape: tuple,
                 backbone: str = 'vgg16',
                 load: bool = False):
        super().__init__()

        if load:
            self.model = keras.models.load_model('unetmodel', compile=False)
        else:
            self.model = sm.Unet(backbone,
                                 classes=classes,
                                 activation='softmax',
                                 input_shape=input_shape)

    def __call__(self, x: np.ndarray, *args, **kwargs):
        return self.model(x)

    def predict_gen(self, x: np.ndarray,):
        if x.ndim < 4:
            x = np.expand_dims(x, axis=0)

        return self.model.predict(x)

    def train(self, train_gen, test_gen):
        self.model.compile(
            'Adam',
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score],
        )

        self.model.fit_generator(
            generator=gen_spec(train_gen),
            steps_per_epoch=500,
            epochs=100,
            validation_data=gen_spec(test_gen),
        )

    def evaluate_c(self, gen):
        self.model.compile(loss=sm.losses.bce_jaccard_loss)
        return self.model.evaluate(gen_spec(gen), steps=10)

