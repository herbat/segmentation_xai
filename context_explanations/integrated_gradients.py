from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils import normalize
from context_explanations.utils import perturb_im_tf, confidence_diff_tf, create_baseline


class IntegratedGradients:

    def __init__(self,
                 image: np.ndarray,
                 model,
                 mask_res: tuple,
                 req_class: int,
                 baseline: Tuple[str, float],
                 path_steps: int = 10):

        self.image = image
        self.model = model
        self.req_class = req_class
        self.orig_out = model.predict_gen(image).squeeze()
        self.bl_image = create_baseline(image=image,
                                        mask_class=req_class,
                                        orig_out=self.orig_out,
                                        baseline=baseline)

        self.path_steps = path_steps
        self.mask_res = mask_res

    def get_grad(self, _smap: np.ndarray) -> np.ndarray:
        smap = tf.Variable(np.expand_dims(np.repeat(_smap[:, :, np.newaxis], 3, axis=2), axis=0))

        orig_im_tf = tf.cast(tf.constant(self.image), 'float32')
        bl_im_tf = tf.cast(tf.constant(self.bl_image), 'float32')

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(smap)

            pert_im = perturb_im_tf(smap=smap, image=orig_im_tf, bl_image=bl_im_tf)
            loss = confidence_diff_tf(orig_out=self.orig_out,
                                      model=self.model,
                                      req_class=self.req_class,
                                      pert_im=tf.cast(pert_im, 'float32'),
                                      im=orig_im_tf)

        grad = tape.gradient(loss, smap).numpy().sum(axis=-1)/3

        return grad.squeeze()

    def generate_saliency_map(self) -> np.ndarray:
        smap = np.ones(self.mask_res)
        grads = []
        for step in np.arange(0, 1, 1/self.path_steps):
            grad = self.get_grad(smap * step)
            grads.append(grad)

        result = np.stack(grads).sum(0)

        return normalize(result)




