import numpy as np
import tensorflow as tf

from context_explanations.utils import perturb_im_tf, confidence_diff_tf


class IntegratedGradients:

    def __init__(self,
                 image: np.ndarray,
                 model,
                 req_class: int,
                 bl_image: np.ndarray,
                 orig_out: np.ndarray,
                 path_steps: int = 10):

        self.image = image
        self.model = model
        self.req_class = req_class
        self.bl_image = bl_image
        self.orig_out = orig_out
        self.path_steps = path_steps

    def get_grad(self, _smap: np.ndarray) -> np.ndarray:
        smap = tf.Variable(np.expand_dims(np.repeat(_smap[:, :, np.newaxis], 3, axis=2), axis=0))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(smap)
            orig_im_tf = tf.cast(tf.constant(self.image), 'float32')
            bl_im_tf = tf.cast(tf.constant(self.bl_image), 'float32')

            pert_im = perturb_im_tf(smap=smap, image=orig_im_tf, bl_image=bl_im_tf)
            loss = confidence_diff_tf(orig_out=self.orig_out,
                                      model=self.model,
                                      req_class=self.req_class,
                                      pert_im=pert_im,
                                      im=self.image)

        grad = tape.gradient(loss, smap).numpy().sum(axis=-1)/3

        return grad.squeeze()

    def generate_saliency_map(self):
        smap = np.ones()
        for step in np.arange(0, 1, 1/self.path_steps):
            grad = self.get_grad(smap * step)




