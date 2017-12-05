import collections
import os.path as osp

import chainer
import numpy as np
from PIL import Image
import scipy.io
from .. import data

DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

class BridgeSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array(['non-damage', 'damage'])

    def __init__(self, split='train'):
        self.split = split

        self.files = collections.defaultdict(list)
        for split in ['train', 'validation']:
            imgsets_file = osp.join(DATASET_BRIDGE_DIR, "{}.txt".format(split))
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(DATASET_BRIDGE_DIR, 'images/combined2/', '{}.jpg'.format(did))
                lbl_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_masks/combined2/', '{}.png'.format(did))
                self.files[split].append({
                    'img' : img_file,
                    'lbl' : lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def get_example(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        img = Image.open(img_file)
        # wsize = int(float(img.size[0]) * 0.5)
        # hsize = int(float(img.size[1]) * 0.5)
        # img = img.resize((wsize, hsize))
        lbl_file = data_file['lbl']
        lbl = Image.open(lbl_file)
        # lbl = lbl.resize((wsize, hsize))

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint32)
        lbl = (lbl/255).astype(np.int32)
        if self.rcrop.any() != None:
            img,lbl = self.random_crop(img,lbl, self.rcrop)
        return img, lbl

    def random_crop(self, x, y=np.array([None]), random_crop_size=None):
        w, h = x.shape[1], x.shape[0]
        rangew = (w - random_crop_size[0]) // 2
        rangeh = (h - random_crop_size[1]) // 2
        offsetw = 0 if rangew==0 else np.random.randint(rangew)
        offseth = 0 if rangeh==0 else np.random.randint(rangeh)
        if y.any() != None:
            return x[offseth:offseth+random_crop_size[0], offsetw:offsetw+random_crop_size[1]], \
                    y[offseth:offseth+random_crop_size[0], offsetw:offsetw+random_crop_size[1]]
        else:
            return x[:, offseth:offseth+random_crop_size[0], offsetw:offsetw+random_crop_size[1]]
        

class BridgeSeg(BridgeSegBase):
    def __init__(self, split='train', rcrop=[None, None]):
       super(BridgeSeg, self).__init__(split=split) 
       if len(rcrop) == 2:
           self.rcrop = np.array(rcrop)
       else:
           self.rcrop = np.array([rcrop, rcrop])


    @staticmethod
    def download():
        print("Not implemented yet (could be a good idea)")
