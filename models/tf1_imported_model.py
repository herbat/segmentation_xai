from typing import List, Union
import tensorflow as tf
import numpy as np

assert tf.__version__ >= "2",\
    "Tensorflow 2 is required for this!"


def wrap_frozen_graph(graph_def: tf.compat.v1.GraphDef,
                      inputs: Union[str, List[str]],
                      outputs: List[str]):
    """
    Returns a tensorflow function which uses the graph of Graphdef.
    """
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


class ImportedTF1Graph(tf.Module):
    def __init__(self, path, inputs: str, outputs: List[str], in_shape: tuple):
        super().__init__()
        self.in_shape = in_shape
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(open(path, 'rb').read())
        self.model_function = wrap_frozen_graph(graph_def,
                                                inputs=inputs,
                                                outputs=outputs)

    def __call__(self, x: tf.Tensor):
        out = self.model_function(x)[0]
        out = tf.image.resize(out, self.in_shape)
        return out

    def predict_gen(self, x: np.ndarray):
        print("Inference")
        if x.ndim < 4:
            x = np.expand_dims(x, axis=0)
        out = self.model_function(tf.constant((x * 255).astype('uint8')))[0]
        out = tf.image.resize(out, self.in_shape)
        return out.numpy()

