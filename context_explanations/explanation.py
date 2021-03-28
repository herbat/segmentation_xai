import abc
from typing import Tuple

import numpy as np


class Explanation:

    @abc.abstractmethod
    def get_explanation(self,
                        image: np.ndarray,
                        mask_res: Tuple[int, int],
                        req_class: int,
                        baseline: Tuple[str, float]):

        """
        :param image:
        :param mask_res:
        :param req_class:
        :param baseline:
        :return: np.ndarray
        """
