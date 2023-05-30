import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter


class DataListLogger(object):
    def __init__(self) -> None:
        self.data = defaultdict(lambda: [])


    def clear(self):
        self.data.clear()


    def add(self, tag, val):
        self.data[tag].append(val)


    def analysis(self):
        res = {}
        res_str = []

        for tag, val_list in self.data.items():
            if len(val_list) > 0:
                res[tag + '_mean'] = np.mean(val_list)
                res[tag + '_std'] = np.std(val_list)
                res_str.append(f" {tag}: {res[tag + '_mean']:.3f} +- {res[tag + '_std']:.3f} ")
        res_str = '|'.join(res_str)
        return res, res_str
    


class SummaryLogger(object):
    def __init__(self, log_path : str):
        self.summary = SummaryWriter(log_path)


    def add(self, counter, scalar_dict, prefix = ''):
        for tag, scalar_val in scalar_dict.items():
            self.summary.add_scalar(prefix + tag, scalar_val, counter)

    
    def add_item(self, counter, tag, val, prefix = ''):
        self.summary.add_scalar(prefix + tag, val, counter)


    def close(self):
        self.summary.close()