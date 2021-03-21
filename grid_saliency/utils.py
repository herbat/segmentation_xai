import cv2
import numpy as np
from utils import zero_nonmax


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


def create_baseline(image: np.ndarray, mask_class: int, orig_out: np.ndarray, baseline: tuple,) -> np.ndarray:
    bl_im = generate_baseline_image(image=image, baseline=baseline)
    zerod_out = zero_nonmax(orig_out)
    bl_im[0, zerod_out[:, :, mask_class] > 0, :] = image[0, zerod_out[:, :, mask_class] > 0, :]
    return bl_im


def perturb_im(image: np.ndarray,
               smap: np.ndarray,
               bl_image: np.ndarray) -> np.ndarray:
    # scale and erode smap
    smap = (smap * 255).astype(np.uint8)
    smap_resized = cv2.resize(smap, image.shape[1:-1], interpolation=cv2.INTER_LINEAR)
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
    diff_area = (zerod_out[:, :, class_r] - cur_out[:, :, class_r]) * req_mask
    diff_area[diff_area < 0] = 0
    return np.sum(diff_area) / req_area_size


def choose_random_n(a: np.ndarray, n: int) -> np.ndarray:
    sample = np.random.uniform(a)
    flat = sample.flatten()
    flat.sort()
    thr = flat[n]
    return sample < thr

