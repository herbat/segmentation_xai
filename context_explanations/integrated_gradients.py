from typing import List, Tuple

import numpy as np
import tensorflow as tf

from utils import normalize
from baseline import Baseline
from context_explanations.explanation import Explanation
from context_explanations.utils import perturb_im_tf, confidence_diff_tf, try_baselines


class IntegratedGradients(Explanation):

    def __init__(self,
                 baselines: List[Baseline],
                 path_steps: int = 10):

        self.baselines = baselines
        self.path_steps = path_steps
        self.name = "Integrated Gradients"

    @staticmethod
    def _get_grad(_smap: np.ndarray,
                  image: np.ndarray,
                  model,
                  req_class: int,
                  baseline: Baseline) -> np.ndarray:
        smap = tf.Variable(np.expand_dims(np.repeat(_smap[:, :, np.newaxis], 3, axis=2), axis=0))

        bl_image = baseline.get_default_baseline(image=image, req_class=req_class, orig_out=model.predict_gen(image))

        orig_out = model.predict_gen(image)
        orig_im_tf = tf.cast(tf.constant(image * 255), 'float32')
        bl_im_tf = tf.cast(tf.constant(bl_image * 255), 'float32')

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(smap)

            pert_im = perturb_im_tf(smap=smap, image=orig_im_tf, bl_image=bl_im_tf)
            loss = confidence_diff_tf(orig_out=orig_out,
                                      model=model,
                                      req_class=req_class,
                                      pert_im=tf.cast(pert_im, 'float32'),
                                      im=orig_im_tf)

        grad = tape.gradient(loss, smap).numpy().sum(axis=-1)/3

        return grad.squeeze()

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: Tuple[int, int],
                        req_class: int) -> np.ndarray:
        smap = np.ones(mask_res)
        grads = []
        baseline = try_baselines(mask_res=mask_res,
                                 baselines=self.baselines,
                                 image=image,
                                 model=model,
                                 orig_out=model.predict_gen(image),
                                 req_class=req_class)

        for step in np.arange(0, 1, 1/self.path_steps):
            grad = self._get_grad(_smap=smap * step,
                                  image=image,
                                  model=model,
                                  req_class=req_class,
                                  baseline=baseline)
            grads.append(grad)

        result = np.stack(grads).sum(0)

        return normalize(result)




