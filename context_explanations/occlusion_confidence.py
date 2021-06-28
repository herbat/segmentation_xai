from typing import Tuple, Iterable, List, Optional

import numpy as np

from baseline import Baseline
from context_explanations.explanation import Explanation
from context_explanations.utils import perturb_im, confidence_diff, try_baselines


class OcclusionSufficiency(Explanation):

    def __init__(self,
                 threshold: float,
                 name: Optional[str] = None,
                 top_k: Optional[int] = None,
                 tune_res: Optional[int] = None,
                 upsample_factor: Optional[int] = None):

        self.name = "Occlusion Sufficiency" + (name if name else "")
        self.threshold = threshold
        self.top_k = top_k
        self.tune_res = tune_res
        self.upsample_factor = upsample_factor

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: tuple,
                        req_class: int,
                        baseline: Baseline) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :param baseline:
        :return:
        """

        smap = np.zeros(mask_res)
        conf_diffs = _get_conf_diffs(smap=smap,
                                     model=model,
                                     image=image,
                                     req_class=req_class,
                                     baseline=baseline,
                                     indices=np.ndindex(smap.shape),
                                     values=[1] * len(smap.flatten()))

        # print(req_class, conf_diffs.max() / np.median(conf_diffs))
        if np.median(conf_diffs) / conf_diffs.min() < self.threshold:
            return smap
        min_diff_idx = int(np.argmin(conf_diffs))
        # print("suff:", conf_diffs.max())
        best_idx = list(np.ndindex(smap.shape))[min_diff_idx]
        smap[best_idx] = 1

        if self.tune_res:
            tune_values = np.linspace(0.5, 1, self.tune_res)
            tuned_diffs = _get_conf_diffs(smap=smap,
                                          model=model,
                                          image=image,
                                          baseline=baseline,
                                          req_class=req_class,
                                          indices=[best_idx] * len(tune_values),
                                          values=tune_values)
            losses = tuned_diffs * tune_values
            tuned_best = tune_values[np.argmin(losses)]

            smap = np.zeros(mask_res)
            smap[best_idx] = tuned_best
        elif self.top_k:
            cd_sorted = np.sort(conf_diffs)
            for i in range(self.top_k):
                nb_diff_idx = np.where(cd_sorted[i] == conf_diffs)[0][0]
                next_best_idx = list(np.ndindex(smap.shape))[nb_diff_idx]
                smap[next_best_idx] = conf_diffs[min_diff_idx] / conf_diffs[nb_diff_idx]
        if self.upsample_factor and not self.tune_res:
            smap = np.ones_like(smap) - smap
            smap = increase_resolution(smap=smap,
                                       factor=self.upsample_factor,
                                       model=model,
                                       image=image,
                                       req_class=req_class,
                                       baseline=baseline,
                                       reverse=True)
        return smap


class OcclusionNecessity(Explanation):

    def __init__(self,
                 threshold: float,
                 name: Optional[str] = None,
                 top_k: Optional[int] = None,
                 tune_res: Optional[int] = None,
                 upsample_factor: Optional[int] = None):
        self.name = "Occlusion Necessity" + (name if name else "")
        self.threshold = threshold
        self.top_k = top_k
        self.tune_res = tune_res
        self.upsample_factor = upsample_factor

    def get_explanation(self,
                        model,
                        image: np.ndarray,
                        mask_res: tuple,
                        req_class: int,
                        baseline: Baseline) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :param baseline:
        :return:
        """

        smap = np.ones(mask_res)
        conf_diffs = _get_conf_diffs(smap=smap,
                                     model=model,
                                     image=image,
                                     req_class=req_class,
                                     baseline=baseline,
                                     indices=np.ndindex(smap.shape),
                                     values=[0] * len(smap.flatten()))

        # print(req_class, conf_diffs.max() / np.median(conf_diffs))
        if conf_diffs.max() / np.median(conf_diffs) < self.threshold:
            return np.zeros(mask_res)
        # print("nec:", conf_diffs.max())
        max_diff_idx = int(np.argmax(conf_diffs))
        best_idx = list(np.ndindex(smap.shape))[max_diff_idx]
        smap[best_idx] = 0
        smap = np.ones_like(smap) - smap

        if self.tune_res:
            smap = np.ones_like(smap) - smap
            tune_values = np.linspace(0, 0.5, self.tune_res)
            tuned_diffs = _get_conf_diffs(smap=smap,
                                          model=model,
                                          image=image,
                                          req_class=req_class,
                                          baseline=best_baseline,
                                          indices=[best_idx] * len(tune_values),
                                          values=tune_values)
            losses = tuned_diffs / (np.ones_like(tune_values) - tune_values)
            tuned_best = tune_values[np.argmax(losses)]
            smap = np.zeros(mask_res)
            smap[best_idx] = 1 - tuned_best
        elif self.top_k:
            cd_sorted = np.sort(conf_diffs)
            for i in range(self.top_k):
                nb_diff_idx = np.where(cd_sorted[-i] == conf_diffs)[0][0]
                next_best_idx = list(np.ndindex(smap.shape))[nb_diff_idx]
                smap[next_best_idx] = conf_diffs[nb_diff_idx] / conf_diffs[max_diff_idx]
        if self.upsample_factor and not self.tune_res:
            smap = increase_resolution(smap=smap,
                                       factor=self.upsample_factor,
                                       model=model,
                                       image=image,
                                       req_class=req_class,
                                       baseline=baseline,
                                       reverse=True)

        return smap


def _get_conf_diffs(smap: np.ndarray,
                    model,
                    image: np.ndarray,
                    req_class: int,
                    baseline: Baseline,
                    indices: Iterable,
                    values: Iterable) -> np.ndarray:

    orig_out = model.predict_gen(image)

    bl_image = baseline.get_default_baseline(image=image, req_class=req_class, orig_out=orig_out)

    conf_diffs = []
    for i, v in zip(indices, values):
        tmp = smap[i]
        smap[i] = v
        pert_im = perturb_im(image=image, smap=smap, bl_image=bl_image)
        cur_out = model.predict_gen(pert_im)
        conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=req_class)
        conf_diffs.append(conf_diff)
        smap[i] = tmp

    conf_diffs = np.stack(conf_diffs)
    return conf_diffs


def increase_resolution(smap,
                        factor,
                        model,
                        image: np.ndarray,
                        req_class: int,
                        baseline: Baseline,
                        reverse: bool):
    new_resolution = [x * factor for x in smap.shape]
    new_smap = np.zeros(new_resolution)
    val = 0 if reverse else 1
    for old_idx in np.ndindex(smap.shape):
        if smap[old_idx] == 0:
            continue

        new_idxs = [(old_idx[0]*factor+nidx[0], old_idx[1]*factor+nidx[1]) for nidx in np.ndindex((factor, factor))]

        conf_diffs = _get_conf_diffs(np.ones_like(new_smap),
                                     model=model,
                                     image=image,
                                     req_class=req_class,
                                     baseline=baseline,
                                     values=[val] * len(new_idxs),
                                     indices=new_idxs)

        best_diff = (conf_diffs.argmax() if reverse else conf_diffs.argmin())
        new_smap[new_idxs[best_diff]] = 1

    return new_smap

