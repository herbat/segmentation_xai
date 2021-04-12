from typing import Optional, List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from utils import zero_nonmax
from baseline import Baseline


def perturb_im(image: np.ndarray,
               smap: np.ndarray,
               bl_image: np.ndarray) -> np.ndarray:
    # scale and erode smap
    smap = (smap * 255).astype(np.uint8)
    smap_resized = cv2.resize(smap, image.shape[1:-1][::-1], interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((3, 3), np.uint8)
    smap_eroded = cv2.erode(smap_resized, kernel=kernel)
    smap = np.repeat(np.expand_dims(smap_eroded, axis=0)[:, :, :, np.newaxis],
                     3, 3).astype(np.float64) / 255

    # apply smap and add the request area(we don't want that area to be affected)
    result = image * smap + bl_image * (1 - smap)

    return result


def loss_fn(lm: float,
            smap: np.ndarray,
            cur_out: np.ndarray,
            orig_out: np.ndarray,
            class_r: int) -> float:

    conf_diff = confidence_diff(cur_out=cur_out, orig_out=orig_out, class_r=class_r)

    return lm * np.sum(smap) + conf_diff


def confidence_diff(cur_out: np.ndarray,
                    orig_out: np.ndarray,
                    class_r: int):
    zerod_out = zero_nonmax(orig_out)
    req_area_size = np.count_nonzero(np.round(zerod_out))
    req_mask = np.round(zerod_out[:, :, class_r]).astype(int)
    diff_area = (zerod_out[:, :, class_r] - cur_out[0, :, :, class_r]) * req_mask
    diff_area[diff_area < 0] = 0
    return np.sum(diff_area) / req_area_size


def choose_random_n(a: np.ndarray, n: int, seed: Optional[int]) -> np.ndarray:
    if seed is not None: np.random.seed(seed)
    sample = np.random.uniform(a)
    flat = sample.flatten()
    flat.sort()
    thr = flat[n]
    return sample < thr


def perturb_im_tf(smap: tf.Variable, image: tf.constant, bl_image: tf.constant):

    smap_resized = tf.keras.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(smap)
    smap_eroded = -tf.nn.max_pool2d(-smap_resized, ksize=(3, 3), strides=1, padding='SAME')
    result = image * smap_eroded + bl_image * (1 - smap_eroded)
    return result


def confidence_diff_tf(orig_out: np.ndarray, req_class: int, model, pert_im: tf.Variable, im: tf.constant) -> float:
    zerod_out = zero_nonmax(orig_out)
    req_area_size = np.count_nonzero(np.round(zerod_out))
    mask_np = np.zeros_like(orig_out)
    mask_np[0, :, :, req_class] = np.round(zerod_out[:, :, req_class]).astype(int)
    mask = tf.constant(mask_np)
    diff = tf.reduce_sum(tf.keras.activations.relu(model(im) - model(pert_im)) * mask)
    return diff / req_area_size


def try_baselines(mask_res: Tuple[int, int],
                  baselines: List[Baseline],
                  image: np.ndarray,
                  model,
                  orig_out: np.ndarray,
                  req_class: int) -> Baseline:

    max_overall = 0
    best_baseline = None

    for bl in baselines:
        max_val = 0
        best_val = 0
        for bl_val, bl_image in bl.get_all_baselines(image=image, req_class=req_class, orig_out=orig_out):
            smap_tmp = np.zeros_like(mask_res)
            im_p = perturb_im(image=image, smap=smap_tmp, bl_image=bl_image)

            out = model.predict_gen(im_p)
            cd = confidence_diff(cur_out=out, orig_out=orig_out, class_r=req_class)
            if cd > max_val:
                best_val = bl_val
                max_val = cd
        bl.default_value = best_val
        if max_val > max_overall:
            best_baseline = bl
            max_overall = max_val

    return best_baseline
