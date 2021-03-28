from typing import Type

import numpy as np

from context_explanations.optimizers import MySGD
from context_explanations.explanation import Explanation
from context_explanations.utils import loss_fn, perturb_im, create_baseline


class GridSaliency(Explanation):

    def __init__(self,
                 baseline: str = None,
                 iterations: int = 100,
                 lm: float = 0.02,
                 batch_size: int = 5,
                 momentum: float = 0.5,
                 learning_rate: float = 0.2):

        self.optimizer = MySGD
        self.baseline = baseline
        self.iterations = iterations
        self.lm = lm
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.name = "Grid Saliency"

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: tuple,
                        req_class: int):

        orig_out = model.predict_gen(image)[0]
        baseline_values = [0, 1/4, 1/2, 3/4, 1]
        losses = []

        for bv in baseline_values:
            smap_tmp = np.zeros(mask_res)
            bl_image = create_baseline(image=image,
                                       mask_class=req_class,
                                       orig_out=orig_out,
                                       baseline=(self.baseline, bv))

            im_p = perturb_im(image=image,
                              smap=smap_tmp,
                              bl_image=bl_image)

            out = model.predict_gen(im_p)[0]
            losses.append(loss_fn(lm=self.lm,
                                  smap=smap_tmp,
                                  cur_out=out,
                                  orig_out=orig_out,
                                  class_r=req_class))

        bl_value = baseline_values[int(np.argmin(losses))]
        bl_image = create_baseline(image=image,
                                   mask_class=req_class,
                                   orig_out=orig_out,
                                   baseline=(self.baseline, bl_value))

        smap = np.ones(mask_res) * 0.5

        opt = self.optimizer(image=image,
                             model=model,
                             req_class=req_class,
                             bl_image=bl_image,
                             orig_out=orig_out,
                             lm=self.lm)

        res, final_loss = opt.optimize(_smap=smap,
                                       iterations=self.iterations,
                                       momentum=self.momentum,
                                       batch_size=self.batch_size,
                                       learning_rate=self.learning_rate)

        if final_loss > min(losses):
            res = smap

        return res


