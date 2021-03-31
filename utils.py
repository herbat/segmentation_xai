from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt


class MetricRecorder:

    def __init__(self, metrics: List[callable]):
        self.metrics = metrics
        self.record = []

    def __call__(self, smap: np.ndarray, mask: np.ndarray) -> None:
        metric_values = []
        for metric in self.metrics:
            metric_values.append(metric(smap, mask))
        self.record.append(metric_values)

    def plot(self):
        plt.plot(self.record)
        plt.show()


def zero_nonmax(a: np.ndarray) -> np.ndarray:
    if a.ndim != 4:
        raise ValueError(f"{a.ndim}")
    max_values = np.repeat((np.max(a, axis=-1))[:, :, :, np.newaxis], a.shape[-1], axis=-1)
    return (a * np.array(a == max_values, dtype=np.uint8)).squeeze()


def decode_segmap(a: np.ndarray, color_array: np.ndarray) -> np.ndarray:
    tmpa = np.argmax(zero_nonmax(a), axis=-1)
    tmpa = np.repeat(tmpa[:, :, np.newaxis], 3, axis=2)
    for i, c in enumerate(color_array):
        try:
            tmpa[tmpa[:, :, 0] == i, :] = c
        except ValueError:
            continue
    return tmpa


def smap_dist(smap: np.ndarray, biased_mask: np.ndarray) -> float:
    bm = cv2.resize((biased_mask*255).astype('uint8'), smap.shape)/255
    return float(np.sum((smap-bm)**2))


def cbl(smap: np.ndarray, biased_mask: np.ndarray) -> float:
    bm = cv2.resize((biased_mask*255).astype('uint8'), smap.shape)/255
    if bm.max() == 0 and smap.max() == 0:
        return 1
    elif smap.max() == 0:
        return 0
    smap_rounded = np.ceil(smap)
    correct_pixels = np.sum(smap[smap_rounded == bm])
    all_pixels = np.sum(smap)

    return correct_pixels/all_pixels


def normalize(a: np.ndarray) -> np.ndarray:
    return (a - a.min())/(a.max() - a.min()) if a.min() != a.max() else a
