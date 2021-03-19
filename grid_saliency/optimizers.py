import abc
from typing import Tuple

import numpy as np
import tensorflow as tf
from scipy.optimize import fmin

from utils import zero_nonmax
from grid_saliency.utils import loss_fn, perturb_im, choose_random_n


class Optimizer:

    def __init__(self,
                 image: np.ndarray,
                 model,
                 req_class: int,
                 bl_image: np.ndarray,
                 orig_out: np.ndarray,
                 lm: float):
        self.image = image
        self.model = model
        self.req_class = req_class
        self.bl_image = bl_image
        self.orig_out = orig_out
        self.lm = lm

    @abc.abstractmethod
    def optimize(self, _smap: np.ndarray, **params) -> Tuple[np.ndarray, float]:
        """
        :param _smap: smap to optimize
        :param params:
        :return: final smap and final loss
        """


class FminOptimizer(Optimizer):

    def __init__(self, image: np.ndarray,
                 model,
                 req_class: int,
                 bl_image: np.ndarray,
                 orig_out: np.ndarray,
                 lm: float):

        super().__init__(image=image, model=model, bl_image=bl_image, req_class=req_class, orig_out=orig_out, lm=lm)
        self.losses = []
        self.lm = None

    def optimize(self, _smap: np.ndarray, iters: int = 1000, lm: float = 0.02) -> Tuple[np.ndarray, float]:
        smap = _smap
        self.lm = lm
        return fmin(self.loss, smap.flatten(), maxiter=iters).reshape(smap.shape), self.losses[-1]

    def loss(self, smap):

        p_im = perturb_im(image=self.image,
                          smap=smap.reshape(smap.shape),
                          bl_image=self.bl_image)

        # get the current output for the perturbed image
        cur_out = self.model(p_im)[0]
        loss = loss_fn(lm=self.lm, smap=smap, cur_out=cur_out, orig_out=self.orig_out, class_r=self.req_class)
        self.losses.append(loss)
        return loss


class TfOptimizer(Optimizer):

    def __init__(self, image: np.ndarray,
                 model,
                 req_class: int,
                 bl_image: np.ndarray,
                 orig_out: np.ndarray,
                 lm: float):

        super().__init__(image=image, model=model, bl_image=bl_image, req_class=req_class, orig_out=orig_out, lm=lm)
        self.losses = []
        self.bl_image = bl_image

    def loss(self, smap: np.ndarray) -> float:

        smap_sum = tf.reduce_sum(smap)
        pert_im = perturb_im(image=self.image,
                             smap=smap,
                             bl_image=self.bl_image)

        zerod_out = zero_nonmax(self.orig_out)
        req_area_size = np.count_nonzero(np.round(zerod_out))
        mask = np.zeros_like(self.orig_out)
        mask[:, :, self.req_class] = np.round(zerod_out[:, :, self.req_class]).astype(int)

        confidence_diff = np.maximum(self.model.predict_gen(self.image) - self.model.predict_gen(pert_im), 0) * mask

        return self.lm * smap_sum + np.sum(confidence_diff) / req_area_size

    @staticmethod
    def perturb(smap: tf.Variable, image: tf.constant, bl_image: tf.constant):

        smap_resized = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(smap)
        smap_eroded = -tf.nn.max_pool2d(-smap_resized, ksize=(3, 3), strides=1, padding='SAME')
        result = image * smap_eroded + bl_image * (1 - smap_eroded)
        return result

    def confidence_diff(self, pert_im: tf.Variable, im: tf.constant) -> float:
        zerod_out = zero_nonmax(self.orig_out)
        req_area_size = np.count_nonzero(np.round(zerod_out))
        mask_np = np.zeros_like(self.orig_out)
        mask_np[:, :, self.req_class] = np.round(zerod_out[:, :, self.req_class]).astype(int)
        mask = tf.constant(mask_np)
        diff = tf.reduce_sum(tf.keras.activations.relu(self.model(im) - self.model(pert_im)) * mask)
        return diff / req_area_size

    def optimize(self, _smap: np.ndarray,
                 iterations: int = 100,
                 batch_size: int = 5,
                 learning_rate: float = 0.2,
                 momentum: float = 0.5) -> Tuple[np.ndarray, float]:

        smap = np.copy(_smap)

        grad_map = self.get_grad(smap)
        grad_map_prev = np.copy(grad_map)

        smap_min = np.zeros_like(smap)
        loss_min = 1

        for i in range(iterations):
            # choose a random set of pixels in the saliency space
            choice = choose_random_n(a=smap, n=batch_size)
            smap[choice] += grad_map[choice] * learning_rate

            smap[smap <= 0.2] = 0
            smap[smap > 1] = 1

            cur_grad = - self.get_grad(smap=smap)
            # update gradients
            grad_map[choice] += cur_grad[choice] * (1 - momentum) + grad_map_prev[choice] * momentum
            grad_map_prev = grad_map
            loss = self.loss(smap=smap)
            self.losses.append(loss)

            if loss_min > loss:
                loss_min = loss
                smap_min = np.copy(smap)

        return smap_min, loss_min

    def get_grad(self, smap: np.ndarray) -> np.ndarray:
        orig_im_tf = tf.cast(tf.constant(self.image), 'float32')
        bl_im_tf = tf.cast(tf.constant(self.bl_image), 'float32')
        smap = tf.Variable(np.expand_dims(np.repeat(smap[:, :, np.newaxis], 3, axis=2), axis=0))

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(smap)
            pert_im = self.perturb(smap=smap, image=orig_im_tf, bl_image=bl_im_tf)
            conf_diff = self.confidence_diff(pert_im=tf.cast(pert_im, 'float32'), im=orig_im_tf)

        grad = tape.gradient(conf_diff, smap).numpy().sum(axis=-1)/3
        grad += self.lm * 0.1

        return grad.squeeze()


class MySGD(Optimizer):

    def __init__(self,
                 image: np.ndarray,
                 model,
                 req_class: int,
                 bl_image: np.ndarray,
                 orig_out: np.ndarray,
                 lm: float):

        super().__init__(image=image, model=model, req_class=req_class, bl_image=bl_image, orig_out=orig_out, lm=lm)
        self.losses = []

    def optimize(self, _smap: np.ndarray,
                 momentum: float = 0.5,
                 iterations: int = 100,
                 batch_size: int = 5,
                 learning_rate: float = 0.2,
                 lm: float = 0.02) -> Tuple[np.ndarray, float]:

        smap = np.copy(_smap)

        grad_map = np.ones_like(smap) * 0.01
        grad_map_prev = np.copy(grad_map)

        eps = 0.01

        smap_min = np.zeros_like(smap)
        loss_min = 1

        for i in range(iterations):

            # choose a random set of pixels in the saliency space
            choice = choose_random_n(smap, batch_size)
            smap[choice] += grad_map[choice] * learning_rate

            smap[smap <= 0] = 0
            smap[smap > 1] = 1

            # get the perturbed image
            p_im = perturb_im(image=self.image,
                              smap=smap,
                              bl_image=self.bl_image)

            # get the current output for the perturbed image
            cur_out = self.model.predict_gen(p_im)[0]

            loss = loss_fn(lm=lm,
                           smap=smap,
                           cur_out=cur_out,
                           orig_out=self.orig_out,
                           class_r=self.req_class)

            # autodiff
            smap_e = np.copy(smap)
            smap_e[choice] += eps
            loss_e = loss_fn(lm=lm,
                             smap=smap_e,
                             cur_out=cur_out,
                             orig_out=self.orig_out,
                             class_r=self.req_class)

            cur_grad = - (loss_e - loss) / eps

            if loss_min > loss:
                loss_min = loss
                smap_min = np.copy(smap)

            self.losses.append(loss_e)
            # update gradients
            grad_map[choice] += cur_grad * (1-momentum) + grad_map_prev[choice] * momentum
            grad_map_prev = grad_map

        return smap_min, loss_min

