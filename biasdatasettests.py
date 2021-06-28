import numpy as np
import pickle

from baseline import Baseline
from utils import cbl
from pipeline_config import dataset
from models.unet_sm_model import UnetModel
from context_explanations.grid_saliency_explanation import GridSaliency


def test_cbl():
    mask_res = (4, 4)

    model = UnetModel(classes=11, input_shape=(64, 64, 3), load=True)

    cur_lm = 0.001
    dic = {}
    for images, req_classes, ms in dataset:
        print('.')

        method = GridSaliency(batch_size=1, iterations=100, lm=cur_lm)
        cur_lm += 0.02
        cbl_values = []
        for image, req_class, m in zip(images, req_classes, ms):
            # if req_class != 2:
            #     continue
            image = np.expand_dims(image, axis=0)
            explanation = method.get_explanation(image=image,
                                                 model=model,
                                                 mask_res=mask_res,
                                                 req_class=req_class,
                                                 baseline=Baseline('value', 0))
            # if np.sum(explanation) != 0:
            #     print(explanation)

            cbl_val = cbl(explanation, biased_mask=m['biased_tile'])
            print(cbl_val)
            cbl_values.append(cbl_val)
        if cur_lm > 0.13:
            break
        print(np.mean(np.array(cbl_values)))
        dic[cur_lm] = cbl_values

    pickle.dump(dic, open('cbl_lambda_results.pkl', 'wb'))


def visualize():
    dic = pickle.load(open('cbl_lambda_results.pkl', 'rb'))


if __name__ == "__main__":
    test_cbl()
    # visualize()
