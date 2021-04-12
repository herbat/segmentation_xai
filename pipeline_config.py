from typing import Tuple

import numpy as np
from tensorflow_datasets.image import Cityscapes

from baseline import Baseline
from presenter import Presenter
from models.unet_sm_model import UnetModel
from models.pspnet_sm_model import PSPNetModel
from models.deeplabv3 import DeepLabV3Plus
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
            result.append(np.argmax(np.sum(np.sum(y, axis=0), axis=0)[:-1]))
        return np.asarray(result)

    def get_x(batch_x):
        return np.repeat(batch_x, 3, axis=-1)

    while True:
        bx, by, _ = next(gen)
        yield get_x(bx), get_y(by)


image_size_x = 64
image_size_y = 64
mask_res = (4, 4)
seed = 1
dataset = dataset_generator(gen_texture_mnist(biased_config, 'test'))
models = [
    # UnetModel(classes=11, input_shape=(64, 64, 3), load=True),
    PSPNetModel(classes=11, input_shape=(64, 64, 3)),
    DeepLabV3Plus(64, 64, nclasses=11)
]

baselines = [
    Baseline('value', 0, possible_values=list(np.linspace(0, 1, 5))),
    Baseline('gaussian', 0.1, possible_values=list(np.linspace(0.1, .3, 5)), seed=seed)
]

explanations = [
    OcclusionSufficiency(baselines=baselines, threshold=1.3, tune_res=10),
    OcclusionNecessity(baselines=baselines, threshold=1.3, tune_res=10),
    IntegratedGradients(baselines=baselines),
    GridSaliency(batch_size=4, iterations=100, baselines=baselines, seed=seed)
]

evaluations = [
    proportionality_necessity,
    proportionality_sufficiency
]

presenter = Presenter(plot=False, print_res=True, save_to_file=False)

