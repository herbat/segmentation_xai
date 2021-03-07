import numpy as np
from matplotlib import pyplot as plt

from model import UnetModel
from utils import decode_segmap, cbl, smap_dist
from grid_saliency.grid_saliency_explanation import GridSaliency
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

generator_biased = gen_texture_mnist(biased_config)
generator_unbiased = gen_texture_mnist(unbiased_config)

# model.train(generator, gen_texture_mnist(config, 'test'))

# print(model.evaluate(generator_biased))
# print(model.evaluate(generator_unbiased))

x, y, m = next(generator_biased)
x_in = np.repeat(x, 3, axis=3)[12:13]
biased_tile = m[12]['biased_tile']

m_out = model(x_in).squeeze()

# plt.imshow(decode_segmap(m_out, colors_mnist))
# plt.show()
#
# plt.imshow(biased_tile)
# plt.show()

for i in range(10):

    out, losses = GridSaliency.generate_saliency_map(image=x_in,
                                                     model=model,
                                                     mask_res=(4, 4),
                                                     req_class=2,
                                                     baseline='value',
                                                     batch_size=3,
                                                     iterations=150)

    print(f'CBL: {cbl(out, biased_tile)}, Dist: {smap_dist(out, biased_tile)}')

    plt.imshow(out.squeeze())
    plt.show()
    #
    # plt.plot(losses)
    # plt.show()


