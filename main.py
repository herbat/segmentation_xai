import time
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from models.unet_sm_model import UnetModel

from evaluations import proportionality
from utils import decode_segmap, cbl, smap_dist, MetricRecorder
from context_explanations.optimizers import MySGD, TfOptimizer
from context_explanations.grid_saliency_explanation import GridSaliency
from context_explanations.integrated_gradients import IntegratedGradients
from context_explanations.occlusion_confidence import OcclusionConfidence
from bias_dataset.mnist_generators_simple import gen_texture_mnist
from bias_dataset.configs import biased_config, unbiased_config


size = 64

model = UnetModel(classes=11, input_shape=(size, size, 3), load=True)

colors_mnist = np.asarray([[250, 227, 227],
                           [247, 212, 188],
                           [227, 189, 184],
                           [207, 165, 180],
                           [204, 152, 183],
                           [203, 146, 184],
                           [201, 139, 185],
                           [184, 110, 167],
                           [132, 107, 138],
                           [100, 75, 80],
                           [58, 8, 66]])

generator_biased = gen_texture_mnist(biased_config, 'test')
generator_unbiased = gen_texture_mnist(unbiased_config, 'test')

# model.train(generator, gen_texture_mnist(config, 'test'))

# print(model.evaluate(generator_biased))
# print(model.evaluate(generator_unbiased))

# m_out = model(x_in).squeeze()

# plt.imshow(decode_segmap(m_out, colors_mnist))
# plt.show()
#

# next(generator_biased)

x, y, m = next(generator_biased)

tf_metric_recorder = MetricRecorder([cbl, smap_dist])

my_metric_recorder = MetricRecorder([cbl, smap_dist])

for i in range(10):

    # if i % 10 != 2 and i % 10 != 1:
    #     continue

    x_in = np.repeat(x, 3, axis=3)[i:i + 1]
    biased_tile = m[i]['biased_tile']
    # out_tf = GridSaliency.generate_saliency_map(image=x_in,
    #                                             model=model,
    #                                             optimizer=TfOptimizer,
    #                                             mask_res=(4, 4),
    #                                             req_class=i % 10,
    #                                             baseline='value',
    #                                             batch_size=4,
    #                                             iterations=100)
    #
    # print("tf", proportionality(out_tf, x_in, model, req_class=i, baseline=('value', 0)))
    #
    # out_my = GridSaliency.generate_saliency_map(image=x_in,
    #                                             model=model,
    #                                             optimizer=MySGD,
    #                                             mask_res=(4, 4),
    #                                             req_class=i % 10,
    #                                             baseline='value',
    #                                             batch_size=5,
    #                                             iterations=100)
    #
    # print("my", proportionality(out_my, x_in, model, req_class=i, baseline=('value', 0)))
    #

    integrated_gradients = IntegratedGradients(image=x_in,
                                               model=model,
                                               mask_res=(4, 4),
                                               req_class=i % 10,
                                               baseline=('value', 0))

    out_ig = integrated_gradients.generate_saliency_map()
    print(proportionality(out_ig, x_in, model, req_class=i, baseline=('value', 0)))

    occlusion_confidence = OcclusionConfidence(image=x_in,
                                               model=model,
                                               mask_res=(4, 4),
                                               req_class=i % 10,
                                               baseline=('value', 0))

    out_oc = occlusion_confidence.generate_saliency_map_sufficiency(threshold=0.02)
    print(proportionality(out_oc, x_in, model, req_class=i, baseline=('value', 0)))
    # tf_metric_recorder(out_tf, biased_tile)
    #
    # my_metric_recorder(out_my, biased_tile)
    #
    # plt.imshow(out_tf, vmin=0, vmax=1)
    # plt.show()
    #
    # plt.imshow(out_my, vmin=0, vmax=1)
    # plt.show()

    plt.imshow(out_ig)
    plt.show()

tf_metric_recorder.plot()

my_metric_recorder.plot()

