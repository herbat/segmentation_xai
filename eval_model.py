
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from keras.utils import generic_utils
from models.tf1_imported_model import ImportedTF1Graph
import sklearn.metrics as mt
import segmentation_models as sm

ds = tfds.load("cityscapes")['train'].batch(1)

def gen():
    for x in ds:
        yield x

b1 = next(gen())
x = b1["image_left"]
x = tf.cast(x, 'uint8')
model = ImportedTF1Graph('deeplabfrozenmodel/deeblab_xc65.pb', "ImageTensor:0", ["SemanticPredictions:0"], (1024, 2048))
out = model(x).numpy().astype('uint8')
gt = b1["segmentation_label"].numpy()

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(out.squeeze())
ax2 = fig.add_subplot(132)
ax2.imshow(gt.squeeze())
ax3 = fig.add_subplot(133)
ax3.imshow(x.numpy().squeeze())
fig.savefig('temp.png')

print(pred.dtype, out.dtype)
print(sm.metrics.f1_score(pred/255, out/255))
