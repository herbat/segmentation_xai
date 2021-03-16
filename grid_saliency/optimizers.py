import numpy as np
import tensorflow as tf
from scipy.optimize import fmin
import segmentation_models as sm

from utils import zero_nonmax
from grid_saliency.utils import loss_fn, perturb_im, choose_random_n


class FminOptimizer:

    def __init__(self, image: np.ndarray,
                 model,
                 req_class: int,
                 baseline: str,
                 bl_value: float,
                 orig_out: np.ndarray,
                 lm: float,
                 smap_size: tuple):

        self.image = image
        self.model = model
        self.req_class = req_class
        self.baseline = baseline
        self.bl_value = bl_value
        self.orig_out = orig_out
        self.lm = lm
        self.smap_size = smap_size
        self.losses = []

    def optimize(self, smap: np.ndarray, iters: int):
        return fmin(self.loss, smap.flatten(), maxiter=iters).reshape(self.smap_size)

    def loss(self, smap):
        p_im = perturb_im(im=self.image,
                          smap=smap.reshape(self.smap_size),
                          mask_class=self.req_class,
                          orig_out=self.orig_out,
                          baseline=(self.baseline, self.bl_value))

        # get the current output for the perturbed image
        cur_out = self.model(p_im)[0]
        loss = loss_fn(lm=self.lm, smap=smap, cur_out=cur_out, orig_out=self.orig_out, class_r=self.req_class)
        self.losses.append(loss)
        return loss


class TfOptimizer:

    def __init__(self, orig_im, model, baseline, class_r, lm, orig_out):
        self.orig_im = orig_im
        self.orig_out = orig_out
        self.model = model
        self.baseline = baseline
        self.class_r = class_r
        self.lm = lm
        self.losses = []

    def loss(self, smap: tf.Variable) -> float:

        smap_sum = tf.reduce_sum(smap)
        pert_im = perturb_im(im=self.orig_im,
                             smap=smap.value().numpy(),
                             mask_class=self.class_r,
                             orig_out=self.orig_out,
                             baseline=self.baseline)

        zerod_out = zero_nonmax(self.orig_out)
        req_area_size = np.count_nonzero(np.round(zerod_out))
        mask_np = np.zeros_like(self.orig_out)
        mask_np[:, :, self.class_r] = np.round(zerod_out[:, :, self.class_r]).astype(int)
        mask = tf.cast(tf.constant(mask_np), 'float64')

        confidence_diff = tf.cast(tf.keras.activations.relu(
            self.model(self.orig_im) - self.model(pert_im)), 'float64') * mask

        res = self.lm * smap_sum + tf.reduce_sum(confidence_diff) / req_area_size
        self.losses.append(res)

        return res

    def confidence_diff(self, pert_im: tf.Variable, im: tf.constant):
        zerod_out = zero_nonmax(self.orig_out)
        req_area_size = np.count_nonzero(np.round(zerod_out))
        mask_np = np.zeros_like(self.orig_out)
        mask_np[:, :, self.class_r] = np.round(zerod_out[:, :, self.class_r]).astype(int)
        mask = tf.constant(mask_np)
        diff = tf.reduce_sum(tf.keras.activations.relu(self.model(im) - self.model(pert_im))) * mask
        return diff / req_area_size

    def optimize(self, smap: np.ndarray,
                 iterations: int = 100,
                 batch_size: int = 5,
                 learning_rate: float = 0.2,
                 momentum: float = 0.5):

        smap_var = tf.Variable(smap)

        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss(smap=smap_var)

        grad_map = tape.gradient(loss, smap_var).numpy()
        print(grad_map)
        grad_map_prev = np.copy(grad_map)

        for i in range(iterations):
            # choose a random set of pixels in the saliency space
            choice = choose_random_n(smap, batch_size)
            smap[choice] += grad_map[choice] * learning_rate

            smap[smap <= 0] = 0
            smap[smap > 1] = 1

            smap_var.assign(smap)
            cur_grad = - tape.gradient(loss, smap_var).numpy()
            # update gradients
            grad_map[choice] += cur_grad[choice] * (1 - momentum) + grad_map_prev[choice] * momentum
            grad_map_prev = grad_map

        return smap_var.numpy()

    def get_grad(self, smap: np.ndarray):
        pert_im = perturb_im(im=self.orig_im,
                             smap=smap,
                             mask_class=self.class_r,
                             orig_out=self.orig_out,
                             baseline=self.baseline)
        orig_im_tf = tf.cast(tf.constant(self.orig_im), 'float32')
        pert_im_tf = tf.cast(tf.Variable(pert_im), 'float32')

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(pert_im_tf)
            conf_diff = self.confidence_diff(pert_im=pert_im_tf, im=orig_im_tf)

        return tape.gradient(conf_diff, pert_im_tf)


class MySGD:

    def __init(self):
        pass

