from typing import Tuple

import numpy as np
from tensorflow import keras

from grid_saliency.utils import perturb_im, confidence_diff, create_baseline


def proportionality(smap: np.ndarray,
                    image: np.ndarray,
                    model: keras.Model,
                    baseline: Tuple[str, float],
                    req_class: int) -> float:

    if np.sum(smap) == 0:
        print("Empty smap: proportionality is undefined for an empty saliency map.")
        return np.NAN

    orig_out = model.predict_gen(image).squeeze()
    bl_image = create_baseline(image=image, mask_class=req_class, orig_out=orig_out, baseline=baseline)
    pert_im = perturb_im(image=image, smap=smap, bl_image=bl_image)
    cur_out = model.predict_gen(pert_im).squeeze()
    conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=req_class)
    return conf_diff / np.sum(smap)

