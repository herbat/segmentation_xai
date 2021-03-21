import unittest
import numpy as np

from matplotlib import pyplot as plt

from utils import MetricRecorder, cbl, smap_dist
from bias_dataset.configs import unbiased_config
from bias_dataset.mnist_generators_simple import gen_texture_mnist
from context_explanations.grid_saliency_explanation import loss_fn, perturb_im


class TestLoss(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_out = np.zeros((3, 3, 2))
        self.orig_out[:, :, 1] = np.identity(3)
        self.im = np.zeros_like(self.orig_out)
        self.im[0, 0, 1] = 1
        self.model = lambda x: x
        self.class_r = 1
        self.smap = np.zeros_like(self.orig_out)
        self.smap[2, 2, 1] = 1

    def test_dummy(self):
        expected_out = 1 + 2/3
        actual_out = loss_fn(lm=1,
                             smap=self.smap,
                             cur_out=self.model(self.im),
                             orig_out=self.orig_out,
                             class_r=self.class_r)

        self.assertEqual(actual_out, expected_out, f"Actual out is wrong: {actual_out}")


class TestPerturbation(unittest.TestCase):

    def setUp(self) -> None:
        i, l, m = next(gen_texture_mnist(unbiased_config))
        self.image = i[:1]
        self.image = np.repeat(self.image, 3, 3)
        self.class_r = 0
        self.smap = np.identity(4) * 0.9
        self.orig_out = l[0]

    def test_real(self):
        actual_out = perturb_im(im=self.image,
                                smap=self.smap,
                                mask_class=self.class_r,
                                orig_out=self.orig_out)[0]
        plt.imshow(actual_out)
        plt.show()


class TestMetricRecorder(unittest.TestCase):

    def setUp(self) -> None:
        self.metricrecorder = MetricRecorder([cbl, smap_dist])

    def test_recording(self):
        self.metricrecorder(np.random.randn(4, 4), np.random.randn(4, 4))

        self.metricrecorder(np.random.randn(4, 4), np.random.randn(4, 4))

        self.metricrecorder(np.random.randn(4, 4), np.random.randn(4, 4))

        self.metricrecorder.plot()
