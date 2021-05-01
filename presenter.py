import pickle
import numpy as np
from matplotlib import pyplot as plt


class Presenter:

    def __init__(self, plot: bool = False, print_res: bool = False, save_to_file: bool = False):
        self.plot = plot
        self.print_res = print_res
        self.save_to_file = save_to_file

    def __call__(self, data_fname: str, *args, **kwargs):
        data = pickle.load(open(data_fname, "rb"))

        suf = []
        nec = []
        names = ["Oc-Suf full", "Oc-nec full", "Oc-Suf top 4", "Oc-nec top 4", "Oc-suf", "Oc-nec"]
        for n, d in data.items():
            if self.plot:
                plt.plot(d)
                plt.title(n)
                plt.show()
            if self.print_res:
                if isinstance(d, int): print(n, d)
                else:
                    d = np.asarray(d)
                    res = []
                    for di in d:
                        di = di[~np.isnan(di)]
                        di = di[~np.isinf(di)]
                        res.append(di)
                    nec.append(res[0])
                    suf.append(res[1])
                    # names.append(n)
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(211)
        ax.boxplot(suf, showfliers=False)
        ax.title.set_text("Sufficiency")
        ax.set_xticklabels(names)
        ax2 = fig.add_subplot(212)
        ax2.boxplot(nec, showfliers=False)
        ax2.title.set_text("Necessity")
        ax2.set_xticklabels(names)
        plt.show()


if __name__ == "__main__":
    p = Presenter(False, True, False)
    p(data_fname="./deeplab_apr27_noGS_200_images.pkl")
