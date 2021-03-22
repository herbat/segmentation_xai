from typing import Tuple

import numpy as np
from context_explanations.utils import perturb_im, confidence_diff, create_baseline


class OcclusionConfidence:

    def __init__(self,
                 image: np.ndarray,
                 model,
                 mask_res: tuple,
                 req_class: int,
                 baseline: Tuple[str, float]):

        self.image = image
        self.model = model
        self.req_class = req_class
        self.orig_out = model.predict_gen(image)
        self.bl_image = create_baseline(image=image,
                                        mask_class=req_class,
                                        orig_out=self.orig_out.squeeze(),
                                        baseline=baseline)

        self.mask_res = mask_res

    def generate_saliency_map_sufficiency(self, threshold) -> np.ndarray:
        smap = np.zeros(self.mask_res)
        conf_diffs = []
        for i in np.ndindex(smap.shape):
            smap[i] = 1
            pert_im = perturb_im(image=self.image, smap=smap, bl_image=self.bl_image)
            cur_out = self.model.predict_gen(pert_im)
            conf_diff = confidence_diff(cur_out=cur_out, orig_out=self.orig_out, class_r=self.req_class)
            conf_diffs.append(conf_diff)
            smap[i] = 0

        conf_diffs = np.stack(conf_diffs)
        min_diff = int(np.argmin(conf_diffs))
        smap[list(np.ndindex(smap.shape))[min_diff]] = 1
        return smap

