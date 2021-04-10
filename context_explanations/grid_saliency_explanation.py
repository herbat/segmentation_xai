from typing import List, Optional

import numpy as np

from baseline import Baseline
from context_explanations.optimizers import MySGD
from context_explanations.utils import try_baselines
from context_explanations.explanation import Explanation


class GridSaliency(Explanation):

    def __init__(self,
                 baselines: List[Baseline],
                 iterations: int = 100,
                 lm: float = 0.02,
                 batch_size: int = 5,
                 momentum: float = 0.5,
                 learning_rate: float = 0.2,
                 seed: Optional[int] = None):

        self.optimizer = MySGD
        self.baselines = baselines
        self.iterations = iterations
        self.lm = lm
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.name = "Grid Saliency"
        self.seed = seed

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: tuple,
                        req_class: int):

        orig_out = model.predict_gen(image)
        losses = []

        baseline = try_baselines(mask_res=mask_res,
                                 baselines=self.baselines,
                                 image=image,
                                 model=model,
                                 orig_out=orig_out,
                                 req_class=req_class)

        bl_image = baseline.get_default_baseline(image=image, orig_out=orig_out, req_class=req_class)

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
                                       learning_rate=self.learning_rate,
                                       seed=self.seed)

        if final_loss > min(losses):
            res = np.zeros(mask_res)

        return res


