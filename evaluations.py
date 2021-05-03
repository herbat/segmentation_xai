from typing import Tuple

import numpy as np
from tensorflow import keras

from baseline import Baseline
from context_explanations.utils import perturb_im, confidence_diff


def proportionality_sufficiency(smap: np.ndarray,
                                image: np.ndarray,
                                model: keras.Model,
                                baseline: Baseline,
                                req_class: int) -> tuple:

    """
    We want this to be big!
    :param smap:
    :param image:
    :param model:
    :param baseline:
    :param req_class:
    :return:
    """

    orig_out = model.predict_gen(image)
    bl_image = baseline.get_default_baseline(image=image, req_class=req_class, orig_out=orig_out)
    pert_im = perturb_im(image=image, smap=smap, bl_image=bl_image)
    cur_out = model.predict_gen(pert_im)
    conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=req_class)
    return conf_diff, np.mean(smap)


def proportionality_necessity(smap: np.ndarray,
                              image: np.ndarray,
                              model: keras.Model,
                              baseline: Baseline,
                              req_class: int) -> tuple:
    """
    We want this to be big too!
    :param smap:
    :param image:
    :param model:
    :param baseline:
    :param req_class:
    :return:
    """

    orig_out = model.predict_gen(image)
    bl_image = baseline.get_default_baseline(image=image, req_class=req_class, orig_out=orig_out)
    smap_inv = np.ones_like(smap) - smap
    pert_im = perturb_im(image=image, smap=smap_inv, bl_image=bl_image)
    cur_out = model.predict_gen(pert_im)
    conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=req_class)
    return conf_diff, np.mean(smap)
