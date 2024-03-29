from datetime import datetime
import pickle

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from baseline import Baseline
from pipeline_config import models, dataset, explanations, evaluations, mask_res, seed, baselines
from utils import zero_nonmax
from context_explanations.utils import try_baselines

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
        eval_results = {'seed': seed}
        for i in range(len(explanations)):
            eval_results[explanations[i].name] = []
            for j in range(len(evaluations)):
                eval_results[explanations[i].name].append([])

        batch_count = 0
        for x_batch, y_batch in dataset:
            print(f"{datetime.now().strftime('%m/%d-%H:%M')}: Batch {batch_count} started.")
            batch_count += 1
            for image, req_class in zip(x_batch, y_batch):
                image = np.expand_dims(image, axis=0)
                orig_out = model.predict_gen(image)
                if len(baselines) > 1:
                    baseline = try_baselines(mask_res=mask_res,
                                             model=model,
                                             baselines=baselines,
                                             image=image,
                                             orig_out=orig_out,
                                             req_class=req_class)
                else:
                    baseline = baselines[0]
                if req_class == -1:
                    print("No object of interest on the image, skipping...")
                    continue
                for explanation_method in explanations:
                    explanation = explanation_method.get_explanation(image=image,
                                                                     model=model,
                                                                     mask_res=mask_res,
                                                                     req_class=req_class)

                    # plt.imshow(explanation, vmin=0, vmax=1)
                    # plt.title(explanation_method.name)
                    # plt.show()

                    for i, evaluation in enumerate(evaluations):
                        eval_results[explanation_method.name][i].append(evaluation(smap=explanation,
                                                                                   image=image,
                                                                                   model=model,
                                                                                   req_class=req_class,
                                                                                   baseline=Baseline('value', 0)))
                
            if batch_count > 200: break
        outfile = open(f"{model.name}_{datetime.now().strftime('%mm%dd%Hh%Mm')}.pkl", "wb")
        pickle.dump(eval_results, outfile)
        print(f"File saved successfully.")



