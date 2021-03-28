from typing import Tuple
import numpy as np
from models.unet_sm_model import UnetModel
from bias_dataset.mnist_generators_simple import gen_texture_mnist
from bias_dataset.configs import biased_config, unbiased_config
from context_explanations.grid_saliency_explanation import GridSaliency
from context_explanations.integrated_gradients import IntegratedGradients
from evaluations import proportionality_necessity, proportionality_sufficiency
from context_explanations.occlusion_confidence import OcclusionSufficiency, OcclusionNecessity


def dataset_generator(gen) -> Tuple[np.ndarray, np.ndarray]:
    def get_y(batch_y):
        result = []
        for y in batch_y:
            result.append(np.argmax(np.sum(np.sum(y, axis=0), axis=0)))
        return np.asarray(result)

    def get_x(batch_x):
        return np.repeat(batch_x, 3, axis=-1)

    while True:
        bx, by, _ = next(gen)
        yield get_x(bx), get_y(by)


image_size_x = 64
image_size_y = 64
mask_res = (4, 4)
dataset = dataset_generator(gen_texture_mnist(biased_config, 'test'))
models = [UnetModel(classes=11, input_shape=(64, 64, 3), load=True)]
explanations = [OcclusionSufficiency(baseline=('value', 0)),
                OcclusionNecessity(baseline=('value', 0)),
                IntegratedGradients(baseline=('value', 0)),
                GridSaliency(batch_size=4, iterations=100, baseline='value')]
evaluations = [proportionality_necessity,
               proportionality_sufficiency]
presenter = print

