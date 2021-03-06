# sys
import numpy as np
import pickle
import torch

# operation
from .utils import skeleton


class SkeletonFeeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 duo_only=False,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.duo_only = duo_only
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.duo_only:
            idx = []
            new_label = []
            new_sample_name = []
            for (i, l) in enumerate(self.label):
                # sample only interactions
                if (l >= 49 and l <= 59) or (l >= 105 and l <= 119):
                    if l >= 49 and l <= 59:
                        l = l-49
                    else:
                        l = l-105 + 11
                # sample end
                    idx.append(i)
                    new_label.append(l)
                    new_sample_name.append(self.sample_name[i])
            N, C, T, V, M = self.data.shape
            self.data = self.data[idx]
            print("after data shape: {}".format(self.data.shape))

            self.label = new_label
            self.sample_name = new_sample_name

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print("data shape: {}".format(self.data.shape))
        print("label shape: {}".format(len(self.label)))
        print("sample name shape: {}".format(len(self.sample_name)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        if self.random_choose:
            data_numpy = skeleton.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = skeleton.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = skeleton.random_move(data_numpy)

        return data_numpy, label
