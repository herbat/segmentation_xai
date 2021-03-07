import numpy as np
import cv2


def zero_nonmax(a: np.ndarray) -> np.ndarray:
    return a * (a >= np.sort(a, axis=2)[:, :, [-1]])


def decode_segmap(a: np.ndarray, color_array: np.ndarray) -> np.ndarray:
    tmpa = np.argmax(zero_nonmax(a), axis=-1)
    tmpa = np.repeat(tmpa[:, :, np.newaxis], 3, axis=2)
    for i, c in enumerate(color_array):
        try:
            tmpa[tmpa[:, :, 0] == i, :] = c
        except ValueError:
            continue
    return tmpa


def smap_dist(smap: np.ndarray, biased_mask: np.ndarray):
    bm = cv2.resize((biased_mask*255).astype('uint8'), smap.shape)
    return np.sum((smap-bm)**2)


def cbl(smap: np.ndarray, biased_mask: np.ndarray):
    bm = cv2.resize((biased_mask*255).astype('uint8'), smap.shape)
    smap_rounded = np.ceil(smap)
    correct_pixels = np.sum(smap[smap_rounded == bm])
    all_pixels = np.sum(smap)

    return correct_pixels/all_pixels
