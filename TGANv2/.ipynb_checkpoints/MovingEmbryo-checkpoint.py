from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import pandas as pd
from numpy import asarray

from pathlib import Path

from natsort import natsorted


class MovingEmbryo(data.Dataset):
    
    raw_folder = ''
    processed_folder = ''
    training_file = ''
    test_file = ''


    def __init__(self, 
                 root, 
                 train=True, 
                 split=0, 
                 transform=None, 
                 target_transform=None, 
                 download=False, 
                 process=False):
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set

        if download:
            raise RuntimeError('Embryo Dataset can not be downloaded')

        if process: 
            self.process()

        if self.train:
            self.train_data = torch.load(
                os.path.join(Path(self.root).parent, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(Path(self.root).parent, self.processed_folder, self.test_file))



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """
        seq, target = self.train_data[index, :10], self.train_data[index, 10:]

        return seq, target


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(Path(self.root).parent, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(Path(self.root).parent, self.processed_folder, self.test_file))

    def process(self):
        # process and save as torch files
        print('Processing...')

        numpydata = None

        for folder in sorted(os.listdir(self.root)):
            evol = None
            
            for img_name in natsorted(os.listdir(os.path.join(self.root,folder))):
                f = os.path.join(self.root,folder,img_name)
                if os.path.isfile(f):
                    image = Image.open(f)

                    if evol is None:
                        evol = asarray(image)
                        evol = evol[np.newaxis, :, :]
                    else: 
                        individual = asarray(image)
                        individual = individual[np.newaxis, :, :]
                        evol = np.concatenate((evol, individual), axis=0)
                    
            if numpydata is None:
                numpydata = evol
                numpydata = numpydata[np.newaxis, :, :, :]
            else:
                evol = evol[np.newaxis, :, :, :]
                numpydata = np.concatenate((numpydata, evol), axis=0)

        np.save(os.path.join(Path(self.root).parent, 'moving_embryo_256_20.npy'), numpydata)
        
        training_set = torch.from_numpy(
            np.load(os.path.join(Path(self.root).parent, 'moving_embryo_256_20.npy'))#.swapaxes(0, 1)
        )

        with open(os.path.join(Path(self.root).parent, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Train/test: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
