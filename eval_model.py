
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from models.tf1_imported_model import ImportedTF1Graph
import sklearn.metrics as mt

ds = tfds.load("cityscapes")['train'].batch(100)

gen = [x for x in tfds.as_numpy(ds)]

x = gen[0]["image_left"]

model = ImportedTF1Graph('deeplabfrozenmodel/deeblab_xc65.pb', "ImageTensor:0", ["ResizeBilinear_3:0"], (1024, 2048))

print(mt.jaccard_score(model.predict_gen(x), gen[0]["segmentation_label"]))
