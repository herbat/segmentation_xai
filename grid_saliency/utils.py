import cv2
import numpy as np
import tensorflow as tf
from utils import zero_nonmax
from tensorflow.keras import Model


def generate_baseline_image(image: np.ndarray, baseline: tuple) -> np.ndarray:
    """
    Generate different types of baselines.

    :param np.ndarray image: image to generate baseline for
    :param str baseline: name of the baseline type
    """
    types = {
        'blur': lambda im, x: cv2.GaussianBlur(im, (x, x), 0),
        'value': lambda im, x: np.ones_like(im) * x,
        'uniform': lambda im, x: np.random.uniform(0, x, im.shape),
        'gaussian': lambda im, x: np.abs(np.random.normal(0, x, im.shape))
    }

    return types[baseline[0]](image, baseline[1])


def perturb_im(im: np.ndarray,
               smap: np.ndarray,
               mask_class: int,
               orig_out: np.ndarray,
               baseline: tuple = ('value', 0)) -> np.ndarray:
    # scale and erode smap
    smap = (smap * 255).astype(np.uint8)
    smap_resized = cv2.resize(smap, im.shape[1:-1], interpolation=cv2.INTER_LINEAR)
    kernel = np.ones((3, 3), np.uint8)
    smap_eroded = cv2.erode(smap_resized, kernel=kernel)
    smap = np.repeat(np.expand_dims(smap_eroded, axis=0)[:, :, :, np.newaxis],
                     3, 3).astype(np.float64) / 255

    # generate baseline image
    b_im = generate_baseline_image(im, baseline)

    # apply smap and add the request area(we don't want that area to be affected)
    result = im * smap + b_im * (1 - smap)
    zerod_out = zero_nonmax(orig_out)
    result[0, zerod_out[:, :, mask_class] > 0, :] = im[0, zerod_out[:, :, mask_class] > 0, :]
    return result


def loss_fn(lm: float,
            smap: np.ndarray,
            cur_out: np.ndarray,
            orig_out: np.ndarray,
            class_r: int) -> float:
    zerod_out = zero_nonmax(orig_out)
    req_area_size = np.count_nonzero(np.round(zerod_out))
    req_mask = np.round(zerod_out[:, :, class_r]).astype(int)
    diff_area = (zerod_out[:, :, class_r] - cur_out[:, :, class_r]) * req_mask
    diff_area[diff_area < 0] = 0
    return lm * np.sum(smap) + (np.sum(diff_area) / req_area_size)


def choose_random_n(a: np.ndarray, n: int) -> np.ndarray:
    sample = np.random.uniform(a)
    flat = sample.flatten()
    flat.sort()
    thr = flat[n]
    return sample < thr

