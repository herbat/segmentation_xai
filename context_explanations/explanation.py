import abc
from typing import Tuple

import numpy as np

from baseline import Baseline


class Explanation:

    @abc.abstractmethod
    def get_explanation(self,
                        image: np.ndarray,
                        model,
                        mask_res: Tuple[int, int],
                        req_class: int,
                        baseline: Baseline):

        """
        :param image:
        :param model:
        :param mask_res:
        :param req_class:
        :param baseline:
        :return: np.ndarray
        """
