import numpy as np
from matplotlib import pyplot as plt


class Presenter:

    def __init__(self, plot: bool = False, print_res: bool = False, save_to_file: bool = False):
        self.plot = plot
        self.print_res = print_res
        self.save_to_file = save_to_file

    def __call__(self, model_name: str, data: dict, *args, **kwargs):
        print(model_name)
        for n, d in data.items():
            if self.plot:
                plt.plot(d)
                plt.title(n)
                plt.show()
            if self.print_res:
                print(n, np.average(d))
            if self.save_to_file:
                f = open(f'{model_name}_savefile.txt', 'w')
                f.write(str(n) + ": " + str(d))

