import chainer
import collections
import numpy as np
import os.path as osp
import piexif
import scipy.io

import imgaug as ia
from imgaug import augmenters as iaa

from PIL import Image
from .. import data

#DATASET_BRIDGE_DIR = osp.expanduser('/root/fcn/bridgedegradationseg/dataset/')
DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

class BridgeSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array(['non-damage', 'delamination', 'rebar_exposure'])
    class_weight_default = np.array([0.3610441, 4.6313269, 69.76223605]) #the weights will be multiplied with the loss value

    def __init__(self, split='train', use_data_augmentation=True, black_out_non_deck=False, use_class_weight=False):
        self.split = split
        self.black_out_non_deck = black_out_non_deck
        self.use_data_augmentation = use_data_augmentation
        #if black_out_non_deck or use_data_augmentation:
            #see below with the class weights
            #self.class_names = np.array(['non-damage', 'delamination', 'rebar_exposure', 'non-deck'])
        
        if use_class_weight:
            self.class_weight = BridgeSegBase.class_weight_default
            #if black_out_non_deck:
                #non deck is label -1, should be ignored by the softmax cross entropy
                #so adding an additional weight might fuck up the order if classes
                #lets see if this throws an error or not
                #self.class_weight = np.append(self.class_weight, 0.0)
        else:
            self.class_weight = None

        if self.use_data_augmentation:
            ia.seed(42)
            self.seq = iaa.Sequential([
                iaa.Fliplr(0.5, name='Fliplr'),
                iaa.Flipud(0.5, name='Flipud'),
                iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, rotate=(-45, 45), shear=(-15, 15), name='Affine')),
                iaa.SomeOf((0,1), [ #pick from 0 up to 1 out of the following to apply
                    iaa.ContrastNormalization((0.75, 1.5), name='ContrastNormalization'),     
                    iaa.Multiply((0.8, 1.2), name='Multiply')
                ], random_order=True)
            ])
            def activator_lbl(images, augmenter, parents, default):
                if augmenter.name in ['ContrastNormalization', 'Multiply']:  #all these will not be performed on the labels
                    return False
                else:
                    # default value for all other augmenters
                    return default
            self.hooks_lbl = ia.HooksImages(activator=activator_lbl)

        self.files = collections.defaultdict(list)
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
        piexif.remove(img_file)
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

        if self.use_data_augmentation:
            lbl+=1 #imaug library pads with 0. We want the label to be padded with 'non-deck', which has the label -1, hence this cheap hack
            img, lbl = self.augment_image(img,lbl)
            lbl-=1
            #if np.unique(lbl).shape[0] > len(self.class_names):
            if np.unique(lbl).shape[0] > len(self.class_names)+1: #+1 because we add -1 as a label, whcih doesnt have a class name
                print('WARNING: someting is odd about the number of labeled classes in this image, the are {} (label: {})'.format(np.unique(lbl), lbl_file))


        return img, lbl

    def random_crop(self, x, y=np.array([None]), random_crop_size=None):
        w, h = x.shape[1], x.shape[0]
        rangew = (w - random_crop_size[0]) // 2
        rangeh = (h - random_crop_size[1]) // 2
        offsetw = 0 if rangew<=0 else np.random.randint(rangew)
        offseth = 0 if rangeh<=0 else np.random.randint(rangeh)
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

    def augment_image(self, img, lbl):
    
        seq_det = self.seq.to_deterministic() #seq_det is now a fixes sequence, so lbl and deck get treated the same as img
                                         #it will be a new one after each call of seq.to_deterministic()  
        img = seq_det.augment_image(img)
        lbl = seq_det.augment_image(lbl, hooks=self.hooks_lbl)

        return img, lbl

        

class BridgeSeg(BridgeSegBase):
    def __init__(self, split='train', rcrop=[None, None], use_data_augmentation=True, black_out_non_deck=False, use_class_weight=False):
       super(BridgeSeg, self).__init__(split=split, use_data_augmentation=use_data_augmentation, black_out_non_deck=black_out_non_deck, use_class_weight=use_class_weight) 
       if len(rcrop) == 2:
           self.rcrop = np.array(rcrop)
       else:
           self.rcrop = np.array([rcrop, rcrop])


    @staticmethod
    def download():
        print("Not implemented yet (could be a good idea)")
