
import numpy as np
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
        baseline_values = {0, 1/4, 1/2, 3/4, 1}
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
        bl_value = np.argmin(losses)

        # optimize

        smap = np.ones(mask_res) * 0.5
        grad_map = np.ones(mask_res) * 0.1
        grad_map_prev = np.ones(mask_res) * 0.1
        i_losses = []
        prev_loss = np.min(losses)
        i_losses.append(prev_loss)
        for i in range(iterations):

            # choose a random set of pixels in the saliency space
            choice = choose_random_n(smap, batch_size)
            smap[choice] += (grad_map[choice]*(1-momentum) +
                             grad_map_prev[choice]*momentum) * learning_rate

            smap[smap <= 0.2] = 0

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
            i_losses.append(loss)
            # update gradients
            grad_map[choice] += (prev_loss-loss)
            grad_map_prev = grad_map

        if i_losses[0] < i_losses[-1]:
            smap = np.zeros(mask_res)

        return smap, i_losses

