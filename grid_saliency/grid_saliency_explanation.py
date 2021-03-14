import cv2
import numpy as np
from scipy.optimize import fmin
from matplotlib import pyplot as plt
from grid_saliency.utils import loss_fn, perturb_im, choose_random_n


class GridSaliency:

    @staticmethod
    def generate_saliency_map(image: np.ndarray, model,
                              mask_res: tuple,
                              req_class: int,
                              baseline: str = None,
                              iterations: int = 100,
                              lm: float = 0.02,
                              batch_size: int = 5,
                              momentum: float = 0.5,
                              learning_rate: float = 0.2):

        orig_out = model(image)[0]
        baseline_values = [0, 1/4, 1/2, 3/4, 1]
        losses = []
        for bv in baseline_values:
            smap_tmp = np.zeros(mask_res)
            im_p = perturb_im(im=image,
                              smap=smap_tmp,
                              mask_class=req_class,
                              orig_out=orig_out,
                              baseline=(baseline, bv))
            out = model(im_p)[0]
            losses.append(loss_fn(lm=lm,
                                  smap=smap_tmp,
                                  cur_out=out,
                                  orig_out=orig_out,
                                  class_r=req_class))

        bl_value = baseline_values[int(np.argmin(losses))]

        # optimize
        eps = 0.1
        smap = np.ones(mask_res) * 0.5
        grad_map = np.ones(mask_res) * 0.1
        grad_map_prev = np.ones(mask_res) * 0.1
        i_losses = []
        prev_loss = np.min(losses)
        # if prev_loss < 0.03:
        #     return np.zeros(mask_res), []
        i_losses.append(prev_loss)

        smap_min = np.zeros(mask_res)
        loss_min = prev_loss

        for i in range(iterations):

            # choose a random set of pixels in the saliency space
            choice = choose_random_n(smap, batch_size)
            smap[choice] += grad_map[choice] * learning_rate

            smap[smap <= 0] = 0
            smap[smap > 1] = 1

            # get the perturbed image
            p_im = perturb_im(im=image,
                              smap=smap,
                              mask_class=req_class,
                              orig_out=orig_out,
                              baseline=(baseline, bl_value))

            # get the current output for the perturbed image
            cur_out = model(p_im)[0]

            loss = loss_fn(lm=lm,
                           smap=smap,
                           cur_out=cur_out,
                           orig_out=orig_out,
                           class_r=req_class)

            # autodiff
            smap_e = np.copy(smap)
            smap_e[choice] += eps
            loss_e = loss_fn(lm=lm,
                             smap=smap_e,
                             cur_out=cur_out,
                             orig_out=orig_out,
                             class_r=req_class)

            cur_grad = - (loss_e - loss) / eps

            if loss_min > loss:
                loss_min = loss
                smap_min = np.copy(smap)

            i_losses.append(loss)
            # update gradients
            grad_map[choice] += cur_grad * (1-momentum) + grad_map_prev[choice] * momentum
            grad_map_prev = grad_map

        if i_losses[0] < loss_min:
            smap = np.zeros(mask_res)
        else:
            smap = smap_min

        return smap, i_losses


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

