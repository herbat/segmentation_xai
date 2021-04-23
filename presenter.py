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
                        res.append(np.average(di))
                    print(n, res)


if __name__ == "__main__":
    p = Presenter(False, True, False)
    p(data_fname="./unet_model_04m16d14h19m.pkl")
