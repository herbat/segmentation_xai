import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from baseline import Baseline
from pipeline_config import dataset_cityscapes
from context_explanations.utils import perturb_im


def filter_nan_inf(a):
    a = a[~np.isnan(a[0])]
    a = a[~np.isinf(a[1])]
    return a


def load_data(data: dict):
    suf = []
    nec = []

    for n, d in data.items():
        if isinstance(d, int):
            print(n, d)
        else:
            nec.append(np.asarray(d[1]))
            print(np.shape(d[1]), np.shape(d[2]))
            # print(d[1], d[2])
            suf.append(np.asarray(d[2]))

    return np.stack(nec), np.stack(suf)


colors_n = ['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:brown', 'tab:olive']
colors_a = ['lightblue', 'bisque', 'tomato', 'lightgreen', 'plum', 'chocolate', 'yellowgreen']


def plot_scatter_allmethods(ax, a, names, legend: bool, factor: int):
    for i in range(0, a.shape[0]):
        ax.scatter(a[i, :, 1]*factor, a[i, :, 0], s=50, marker="o", color=colors_n[i])
    for i in range(0, a.shape[0]):
        ax.scatter(np.average(a[i, :, 1]*factor), np.average(a[i, :, 0]), s=200,
                   marker="x", color=colors_a[i])
    plt.xlabel("Sum of saliency values")
    plt.ylabel("Confidence loss")
    if legend:
        ax.legend(names + ['']*len(names), bbox_to_anchor=(0.2, 1.02, 0.5, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=2, prop={'size': 15})


def plot_allmethods(data):
    nec, suf = load_data(data)
    fig = plt.figure(figsize=(12, 18))
    # gs = plt.GridSpec(1, 3, width_ratios=[2, 2, 1])
    ax0 = fig.add_subplot(2, 1, 1)
    plot_scatter_allmethods(ax0, nec[:-1], names=['Sufficiency, k=48, average:',
                                                  'Necessity, k=48,   average:',
                                                  'Sufficiency, k=4,  average:',
                                                  'Necessity, k=4,    average:',
                                                  'Sufficiency, k=1,  average:',
                                                  'Necessity, k=1,    average:'], legend=True, factor=48)
    ax1 = fig.add_subplot(2, 1, 2)
    # plt.title('Normal explanation shown on image', fontsize=16)
    plot_scatter_allmethods(ax1, suf[:-1], names=['Sufficiency, k=48, average:',
                                                  'Necessity, k=48, average:',
                                                  'Sufficiency, k=4, average:',
                                                  'Necessity, k=4, average:',
                                                  'Sufficiency, k=1, average:',
                                                  'Necessity, k=1, average:'], legend=True, factor=48)
    plt.subplots_adjust(top=0.88, hspace=0.4, bottom=0.05)
    plt.show()


def plot_scatter_comparison(ax, a, names, legend: bool):
    for i in range(0, a.shape[0]):
        ax.scatter(a[i, :, 1]*32, a[i, :, 0], s=50, marker="o", color="lightblue")
    for i in range(0, a.shape[0]):
        ax.scatter(np.average(a[i, :, 1]*32), np.average(a[i, :, 0]), s=200,
                   marker="o", color="red" if i == 0 else "orange")
    plt.xlabel("Sum of saliency values")
    plt.ylabel("Confidence loss")
    if legend:
        ax.legend(names, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                  mode="expand", borderaxespad=0, ncol=2, prop={'size': 15})


def get_baseline_im(image, req_class, gt, baseline: Baseline):
    orig_out = tf.one_hot(gt, depth=21)
    bl_im = baseline.get_default_baseline(image=image, req_class=req_class, orig_out=orig_out)
    return bl_im


def plot_smap_im(data: dict, num: int, ax0, ax1, image, gt, req):
    ax0.imshow(data[num])
    red_im = np.ones_like(image)
    red_im[:, :, :, 1:] = np.zeros_like(image)[:, :, :, 1:]
    # red_im[:, (gt == req).squeeze(), :] = image[:, (gt == req).squeeze(), :]
    print(req)
    pert = perturb_im(image=image, smap=np.ones_like(data[num]) - data[num], bl_image=red_im * 0.5).squeeze()
    pert[(gt == req).squeeze(), :] = image.squeeze()[(gt == req).squeeze(), :]
    ax1.imshow(pert)


def get_n(gen, n):
    res = []
    for i in range(n + 1):
        res.append(next(gen))
    return res


def set_to_zero(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')


def plot_image(data: dict, num: int):
    nec, suf = load_data(data)
    data = data['Occlusion Necessity full']
    d = data[0]
    image, req, gt = get_n(dataset_cityscapes, num)[-1]
    # image = np.expand_dims(np.array(image), axis=0)
    fig = plt.figure(figsize=(18, 12))
    # gs = plt.GridSpec(1, 3, width_ratios=[2, 2, 1])
    ax0 = fig.add_subplot(2, 2, 1)
    fig.suptitle('An example saliency map and its evaluation results', fontsize=28)
    # plt.title('Normal explanation saliency map', fontsize=16)
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 3)

    # plt.title('Normal explanation shown on image', fontsize=16)
    plt.axis('off')
    plot_smap_im(d, num, ax0, ax1, image, gt, req)
    # plt.show()
    ax2 = fig.add_subplot(1, 2, 2)

    # ax2.imshow(get_baseline_im(image=image, req_class=req, gt=gt, baseline=Baseline('value', 0)))
    plot_scatter_comparison(ax2, nec[[1]], ["Our method", "Our method average"], True)
    # plt.title("Necessity confidence loss:")

    ax2.scatter(data[1][num][1] * 32, data[1][num][0])
    plt.subplots_adjust(top=0.78, hspace=0.1, left=0.05, bottom=0.2)
    plt.show()
    # print(req)
    # print(data[1][num], data[2][num])


def plot_many_examples(data):
    data = data['Occlusion Necessity full'][0]
    classes = {
        24: 'person',
        25: 'rider',
        26: 'car',
        27: 'truck',
        28: 'bus',
        31: 'train',
        32: 'motorcycle',
        33: 'bicycle'
    }
    orig = get_n(dataset_cityscapes, 24)
    fig, axes = plt.subplots(6, 4, figsize=(18, 14))
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        image, req, gt = orig[i]
        red_im = np.ones_like(image)
        red_im[:, :, :, 1:] = np.zeros_like(image)[:, :, :, 1:]
        pert = perturb_im(image=image, smap=np.ones_like(data[i]) - data[i], bl_image=red_im * 0.5).squeeze()
        pert[(gt == req).squeeze(), :] = image.squeeze()[(gt == req).squeeze(), :]
        ax.imshow(pert)
        ax.set_title(f'Class: {classes[req[0]]}')

    plt.subplots_adjust(wspace=0.001, left=0.01, right=0.99, top=0.97, bottom=0.01)
    plt.show()


def show_context_baseline():
    image, req, gt = get_n(dataset_cityscapes, 10)[-3]
    baseline = np.random.uniform(size=image.shape).squeeze() * 0.5
    baseline[(gt == req).squeeze(), :] = image.squeeze()[(gt == req).squeeze(), :]
    plt.imshow(baseline)
    plt.axis('off')
    plt.show()


class Presenter:

    def __init__(self, plot: bool = False, print_res: bool = False, save_to_file: bool = False):
        self.plot = plot
        self.print_res = print_res
        self.save_to_file = save_to_file

    def __call__(self, data: dict, *args, **kwargs):
        nec, suf = load_data(data)
        fig = plt.figure(figsize=(8, 6))
        # plt.title("Sufficiency score")
        ax = fig.add_subplot(111)
        plot_scatter(ax, nec[[1, 6]], ["Our method",  names[6], "Our method average", "Grid Saliency average"])
        plt.show()
        # fun(suf[6, :, 1], suf[6, :, 0], suf[1, :, 1], suf[1, :, 0], 30)
        # ax2 = fig.add_subplot(212)
        # plot_scatter(ax2, suf, names)
        # plt.show()

        # ax.boxplot(suf, showfliers=False)
        # ax.title.set_text("Sufficiency")
        # ax.set_xticklabels(names)
        # ax2 = fig.add_subplot(212)
        # ax2.boxplot(nec, showfliers=False)
        # ax2.title.set_text("Necessity")
        # ax2.set_xticklabels(names)
        # plt.show()


def vis():
    arrays = [
        [[1, 0],
         [0, 0]],
        [[0, 1],
         [0, 0]],
        [[0, 0],
         [1, 0]],
        [[0, 0],
         [0, 1]],
        [[0.67, 0.85],
         [0.43, 0.12]],
    ]
    for i in arrays:
        plt.imshow(i)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    show_context_baseline()
    # data_dl1 = pickle.load(open("pickles/deeplab_may3_cs_200im_all.pkl", 'rb'))
    # data_dl2 = pickle.load(open("pickles/imported_tf1_graph_05m21d11h01m.pkl", 'rb'))
    # # p = Presenter(False, True, False)
    # # p(data_fname="pickles/deeplab_may6_pascalvoc_600_all.pkl")
    # plot_image(data_dl1, 1)
    # plot_allmethods(data_dl1)
    # plot_many_examples(data_dl1)
