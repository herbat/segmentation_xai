import argparse

import numpy as np

from pipeline_config import models, dataset, explanations, evaluations, presenter, mask_res

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

if __name__ == "__main__":

    for model in models:

        eval_results = {}
        for i in range(len(explanations)):
            eval_results[explanations[i].name] = []
            for j in range(len(evaluations)):
                eval_results[explanations[i].name].append([])

        for x_batch, y_batch in dataset:

            for image, req_class in zip(x_batch, y_batch):
                image = np.expand_dims(image, axis=0)
                for explanation_method in explanations:
                    explanation = explanation_method.get_explanation(image=image,
                                                                     model=model,
                                                                     mask_res=mask_res,
                                                                     req_class=req_class)

                    for i, evaluation in enumerate(evaluations):
                        eval_results[explanation_method.name][i].append(evaluation(smap=explanation,
                                                                                   image=image,
                                                                                   model=model,
                                                                                   req_class=req_class,
                                                                                   baseline=('value', 0)))

            presenter(model.name, eval_results)


