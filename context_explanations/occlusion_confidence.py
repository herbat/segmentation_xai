from typing import Tuple, Iterable, List

import numpy as np

from baseline import Baseline
from context_explanations.explanation import Explanation
from context_explanations.utils import perturb_im, confidence_diff, try_baselines


class OcclusionSufficiency(Explanation):

    def __init__(self, baselines: List[Baseline], threshold: float, tune_res: int):

        self.baselines = baselines
        self.name = "Occlusion Sufficiency"
        self.threshold = threshold
        self.tune_res = tune_res

    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: tuple,
                        req_class: int) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :return:
        """

        smap = np.zeros(mask_res)
        conf_diffs, best_baseline = _get_conf_diffs(smap=smap,
                                                    model=model,
                                                    image=image,
                                                    req_class=req_class,
                                                    baselines=self.baselines,
                                                    indices=np.ndindex(smap.shape),
                                                    values=[1] * len(smap.flatten()))

        print(req_class, np.median(conf_diffs) / conf_diffs.min())
        if np.median(conf_diffs) / conf_diffs.min() > self.threshold:
            return smap
        min_diff = int(np.argmin(conf_diffs))
        best_idx = list(np.ndindex(smap.shape))[min_diff]
        smap[best_idx] = 1

        tune_values = np.linspace(0.5, 1, self.tune_res)
        tuned_diffs, _ = _get_conf_diffs(smap=smap,
                                         model=model,
                                         image=image,
                                         baselines=[best_baseline],
                                         req_class=req_class,
                                         indices=[best_idx] * len(tune_values),
                                         values=tune_values)
        losses = tuned_diffs * tune_values
        tuned_best = tune_values[np.argmin(losses)]

        smap = np.zeros(mask_res)
        smap[best_idx] = tuned_best
        return smap


class OcclusionNecessity(Explanation):

    def __init__(self, baselines: List[Baseline], threshold: float, tune_res: int):
        self.baselines = baselines
        self.name = "Occlusion Necessity"
        self.threshold = threshold
        self.tune_res = tune_res

    def get_explanation(self,
                        model,
                        image: np.ndarray,
                        mask_res: tuple,
                        req_class: int) -> np.ndarray:
        """

        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :return:
        """

        smap = np.ones(mask_res)
        conf_diffs, best_baseline = _get_conf_diffs(smap=smap,
                                                    model=model,
                                                    image=image,
                                                    req_class=req_class,
                                                    baselines=self.baselines,
                                                    indices=np.ndindex(smap.shape),
                                                    values=[0] * len(smap.flatten()))

        print(req_class, conf_diffs.max() / np.median(conf_diffs))
        if conf_diffs.max() / np.median(conf_diffs) < self.threshold:
            return np.zeros(mask_res)

        max_diff = int(np.argmax(conf_diffs))
        best_idx = list(np.ndindex(smap.shape))[max_diff]
        smap[best_idx] = 0

        tune_values = np.linspace(0, 0.5, self.tune_res)
        tuned_diffs, _ = _get_conf_diffs(smap=smap,
                                         model=model,
                                         image=image,
                                         req_class=req_class,
                                         baselines=[best_baseline],
                                         indices=[best_idx] * len(tune_values),
                                         values=tune_values)
        losses = tuned_diffs / (np.ones_like(tune_values) - tune_values)
        tuned_best = tune_values[np.argmax(losses)]
        smap = np.zeros(mask_res)
        smap[best_idx] = 1 - tuned_best
        return smap


def _get_conf_diffs(smap: np.ndarray,
                    model,
                    image: np.ndarray,
                    req_class: int,
                    baselines: List[Baseline],
                    indices: Iterable,
                    values: Iterable) -> Tuple[np.ndarray, Baseline]:

    orig_out = model.predict_gen(image)

    if len(baselines) > 1:
        baseline = try_baselines(mask_res=smap.shape,
                                 model=model,
                                 baselines=baselines,
                                 image=image,
                                 orig_out=orig_out,
                                 req_class=req_class)
    else:
        baseline = baselines[0]

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
    return conf_diffs, baseline
