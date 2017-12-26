import collections
import os.path as osp

import chainer
import numpy as np
from PIL import Image
import scipy.io
from .. import data

DATASET_BRIDGE_DIR = osp.expanduser('/root/fcn/bridgedegradationseg/dataset/')
#DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

class BridgeSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array(['non-damage', 'delamination', 'rebar_exposure'])
    #TODO: figure out proper weights
    class_weight_default = np.array([0.5, 2.0, 4.0]) #the weights will be multiplied with the loss value

    def __init__(self, split='train', black_out_non_deck=False, use_class_weight=False):
        self.split = split
        self.black_out_non_deck = black_out_non_deck
        if black_out_non_deck:
            self.class_names = np.array(['non-damage', 'delamination', 'rebar_exposure', 'non-deck'])

        self.files = collections.defaultdict(list)
        if use_class_weight:
            self.class_weight = class_weight_default
            if black_out_non_deck:
                self.class_weight = np.append(self.class_weight, 0.0)
        else:
            self.class_weight = None

        for split in ['train', 'validation', 'all']:
            imgsets_file = osp.join(DATASET_BRIDGE_DIR, "{}.txt".format(split))
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_dataset/', '{}.jpg'.format(did))
                lbl_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_masks/', '{}.png'.format(did))
                deck_file = ''
                if self.black_out_non_deck:
                    deck_file = osp.join(DATASET_BRIDGE_DIR, 'deck_masks/', '{}.png'.format(did))

                self.files[split].append({
                    'img' : img_file,
                    'lbl' : lbl_file,
                    'deck' : deck_file,
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
        lbl = self.color_class_label(lbl)

        if self.black_out_non_deck:
            deck_file = data_file['deck']
            deck = Image.open(deck_file)
            deck = np.array(deck, dtype=np.uint32)
            img, lbl = self.black_out_non_deck_fn(img, lbl, deck)

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

    def color_class_label(self, image):
        # https://stackoverflow.com/a/33196320
        color_codes = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (255, 0, 0): 2
        }
        
        color_map = np.ndarray(shape=(256*256*256), dtype='int32')
        color_map[:] = -1
        for rgb, idx in color_codes.items():
            rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            color_map[rgb] = idx
        
        image = image.dot(np.array([65536, 256, 1], dtype='int32'))
        return color_map[image]


    def black_out_non_deck_fn(self, img, lbl, deck):
    	assert deck.shape[0:2] == img.shape[0:2]
    	assert img.shape[2] == 3
    	assert len(deck.shape) == 2
        assert lbl.shape == deck.shape
    	deck = deck/255  #so that deck is 1 or 0
        lbl[deck==0] = -1  #make everything none deck as class -1
    	deck = np.repeat(deck[:,:,np.newaxis], 3, axis=2)  #duplicate deck into the 3rd dimension
    	img = img * deck.astype('uint8') 
    	return img, lbl

        

class BridgeSeg(BridgeSegBase):
    def __init__(self, split='train', rcrop=[None, None], black_out_non_deck=False, use_class_weight=False):
       super(BridgeSeg, self).__init__(split=split, black_out_non_deck=black_out_non_deck, use_class_weight=use_class_weight) 
       if len(rcrop) == 2:
           self.rcrop = np.array(rcrop)
       else:
           self.rcrop = np.array([rcrop, rcrop])


    @staticmethod
    def download():
        print("Not implemented yet (could be a good idea)")
