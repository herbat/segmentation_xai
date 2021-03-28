from typing import Tuple

import numpy as np

from context_explanations.explanation import Explanation
from context_explanations.utils import perturb_im, confidence_diff, create_baseline


class OcclusionSufficiency(Explanation):

    def __init__(self,
                 baseline: Tuple[str, float]):

        self.baseline = baseline
        self.name = "Occlusion Sufficiency"

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: tuple,
                        req_class: int) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :return:
        """

        smap = np.zeros(mask_res)
        conf_diffs = _get_conf_diffs(smap=smap,
                                     model=model,
                                     image=image,
                                     req_class=req_class,
                                     baseline=self.baseline,
                                     reverse=False)
        # if conf_diffs.max() < threshold:
        #     return smap
        min_diff = int(np.argmin(conf_diffs))
        smap[list(np.ndindex(smap.shape))[min_diff]] = 1
        return smap


class OcclusionNecessity(Explanation):

    def __init__(self,
                 baseline: Tuple[str, float]):
        self.baseline = baseline
        self.name = "Occlusion Necessity"

    def get_explanation(self,
                        model,
                        image: np.ndarray,
                        mask_res: tuple,
                        req_class: int) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :return:
        """

        smap = np.ones(mask_res)
        conf_diffs = _get_conf_diffs(smap=smap,
                                     model=model,
                                     image=image,
                                     req_class=req_class,
                                     baseline=self.baseline,
                                     reverse=True)
        # if conf_diffs.max() < threshold:
        #     return smap
        max_diff = int(np.argmax(conf_diffs))
        smap = np.zeros_like(smap)
        smap[list(np.ndindex(smap.shape))[max_diff]] = 1
        return smap


def _get_conf_diffs(smap: np.ndarray,
                    model,
                    image: np.ndarray,
                    req_class: int,
                    baseline: Tuple[str, float],
                    reverse: bool):

    orig_out = model.predict_gen(image)
    bl_image = create_baseline(image=image,
                               mask_class=req_class,
                               orig_out=orig_out.squeeze(),
                               baseline=baseline)

    conf_diffs = []
    for i in np.ndindex(smap.shape):
        smap[i] = 0 if reverse else 1
        pert_im = perturb_im(image=image, smap=smap, bl_image=bl_image)
        cur_out = model.predict_gen(pert_im)
        conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=req_class)
        conf_diffs.append(conf_diff)
        smap[i] = 1 if reverse else 0

    conf_diffs = np.stack(conf_diffs)
    return conf_diffs
