from collections import Counter
from typing import Tuple, Type, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from baseline import Baseline
from presenter import Presenter
from models.unet_sm_model import UnetModel
from models.pspnet_sm_model import PSPNetModel
from models.tf1_imported_model import ImportedTF1Graph
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


def cityscapes_generator(shape):
    ds = tfds.load("cityscapes")['train']

    ds = ds.batch(1)
    
    classes_to_check = [24, 25, 26, 27, 28, 31, 32, 33]

    for x in tfds.as_numpy(ds):
        cl, cnt = np.unique(x["segmentation_label"], return_counts=True)
        sums = dict((c, n) for c, n  in zip(cl, cnt) if c in classes_to_check)
        if len(sums.keys()) == 0:
            req_index = -1
        else:
            req_index = list(range(11, 19))[classes_to_check.index(max(sums, key=sums.get))]
        yield tf.image.resize(x['image_left'], shape).numpy() / 255, [req_index]

image_size_x = 512
image_size_y = 1024
mask_res = (4, 8)
seed = 20
dataset = cityscapes_generator([image_size_x, image_size_y])
models = [
    # UnetModel(classes=11, input_shape=(image_size_x, image_size_y, 3), load=True),
    # PSPNetModel(classes=66, input_shape=(image_size_x, image_size_y, 3)),
    # DeepLabV3Plus(64, 64, nclasses=11),
    ImportedTF1Graph('deeplabfrozenmodel/deeblab_xc65.pb', "ImageTensor:0", ["ResizeBilinear_3:0"], (image_size_x, image_size_y))
]

baselines = [
    Baseline('value', 0, possible_values=list(np.linspace(0, 1, 5))),
    Baseline('gaussian', 0.1, possible_values=list(np.linspace(0.1, .3, 5)), seed=seed)
]

explanations = [
    # OcclusionSufficiency(baselines=baselines, threshold=1.2, top_k=32, name=" full"),
    # OcclusionNecessity(baselines=baselines, threshold=1.3, top_k=32, name=" full"),
    # OcclusionSufficiency(baselines=baselines, threshold=1.2, top_k=4, name=" top-4"),
    OcclusionNecessity(baselines=baselines, threshold=1.3, top_k=4, name=" top-4"),
    OcclusionSufficiency(baselines=baselines, threshold=1),
    # OcclusionNecessity(baselines=baselines, threshold=1.3),
    # IntegratedGradients(baselines=baselines),
    
    GridSaliency(batch_size=1, iterations=100, baselines=baselines, seed=seed)
]

evaluations = [
    proportionality_necessity,
    proportionality_sufficiency
]
 

