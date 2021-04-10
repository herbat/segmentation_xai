from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

from context_explanations.utils import zero_nonmax


class Baseline:
    def __init__(self, bl_type: str, default_value: float, possible_values: Optional[List[float]] = None):
        self.bl_type = bl_type
        self.default_value = default_value
        self.possible_values = possible_values

    @staticmethod
    def _generate_baseline_image(image: np.ndarray, baseline: tuple) -> np.ndarray:
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

    @staticmethod
    def _create_baseline(image: np.ndarray, req_class: int, orig_out: np.ndarray, baseline: tuple, ) -> np.ndarray:
        bl_im = Baseline._generate_baseline_image(image=image, baseline=baseline)
        zerod_out = zero_nonmax(orig_out)
        bl_im[0, zerod_out[:, :, req_class] > 0, :] = image[0, zerod_out[:, :, req_class] > 0, :]
        return bl_im

    def get_default_baseline(self, image: np.ndarray, req_class: int, orig_out: np.ndarray) -> np.ndarray:
        return self._create_baseline(image=image,
                                     req_class=req_class,
                                     orig_out=orig_out,
                                     baseline=(self.bl_type, self.default_value))

    def get_all_baselines(self,
                          image: np.ndarray,
                          req_class: int,
                          orig_out: np.ndarray) -> Generator[Tuple[float, np.ndarray], None, None]:

        for value in self.possible_values:
            bl_image = self._create_baseline(image=image,
                                             req_class=req_class,
                                             orig_out=orig_out,
                                             baseline=(self.bl_type, value))
            yield value, bl_image

