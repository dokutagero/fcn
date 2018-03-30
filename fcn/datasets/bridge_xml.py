import chainer
import collections
import numpy as np
import os.path as osp
from os import listdir
import piexif
import scipy.io
import cv2
import random

import imgaug as ia
from imgaug import augmenters as iaa

from PIL import Image
from .. import data

from scripts import label2mask as l2m
from scripts import deck2mask as d2m

DATASET_BRIDGE_DIR = osp.expanduser('/root/bridge_dataset/')
# DATASET_BRIDGE_DIR = osp.expanduser('/home/dokutagero/repos/dataset_bridge/')

class BridgeSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array(['non-damage', 'delamination', 'rebar_exposure'])
    class_weight_default = np.array([0.3610441, 4.6313269, 69.76223605]) #the weights will be multiplied with the loss value

    def __init__(self, split='train', tstrategy=0,  use_data_augmentation=False, black_out_non_deck=False, use_class_weight=False, preprocess=False):
        self.split = split
        self.black_out_non_deck = black_out_non_deck
        self.use_data_augmentation = use_data_augmentation
        self.preprocess = preprocess
        self.tstrategy = tstrategy
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
        for split in ['train_xml', 'validation_xml', 'all']:
            imgsets_file = osp.join(DATASET_BRIDGE_DIR, "{}.txt".format(split))
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_images/', '{}.jpg'.format(did))
                lbl_files = [did.split('/')[-2]+'/'+f for f in listdir(osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', did.split('/')[-2])) if f.startswith(did.split('/')[-1])]
                deck_files = ''

                if self.black_out_non_deck:
                    # deck_file = osp.join(DATASET_BRIDGE_DIR, 'deck_masks/', '{}.png'.format(did))
                    deck_files = [did.split('/')[-2]+'/'+f for f in listdir(osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', did.split('/')[-2])) if f.startswith(did.split('/')[-1])]

                # print(deck_files)
                self.files[split].append({
                    'img' : img_file,
                    'lbl' : lbl_files,
                    'deck' : deck_files,
                })

    def __len__(self):
        return len(self.files[self.split])

    def get_example(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        # piexif.remove(img_file)
        img = Image.open(img_file)
        imsize = img.size
        # wsize = int(float(img.size[0]) * 0.5)
        # hsize = int(float(img.size[1]) * 0.5)
        # img = img.resize((wsize, hsize))
        if self.split == 'xval':
            if not self.tstrategy = 0:
                lbl_file = random.choice(data_file['lbl']) 

            lbl_name = osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', lbl_file)
            lbl = l2m(lbl_name, imsize)

            img = np.array(img, dtype=np.uint8)
            lbl = np.array(lbl, dtype=np.uint32)
            lbl = self.color_class_label(lbl)

            #important: keep this BEFORE black_out_non_deck, as the histogram spreading sometimes causes the black area not to be fully black anymore
            if self.preprocess:
                img = self.preprocess_fn(img)

            if self.black_out_non_deck:
                lbl_names = []
                imsize = img.size
                lbl_names.append(osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', label_file))
                decks = [np.array(d2m(d, imsize)).astype(dtype=np.uint32) for d in lbl_names]
                deck = deck_intersection(decks)
                # deck = np.zeros(decks[0].shape)
                # deck = sum(decks)[:,:,0]
                # deck[deck>0] = 255
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

        elif self.split == 'validation_xml':
            lbl_names = []
            imsize = img.size
            img = np.array(img, dtype=np.uint8)
            for label_file in data_file['lbl']:
                lbl_names.append(osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', label_file))
            masks = [self.color_class_label(l2m(m, imsize)) for m in lbl_names]
            lbl = self.mask_preprocess(masks) 
            if self.black_out_non_deck:
                decks = [np.array(d2m(d, imsize)).astype(dtype=np.uint32) for d in lbl_names]
                deck = np.zeros(decks[0].shape)
                deck = sum(decks)[:,:,0]
                deck[deck>0] = 255
                img, lbl = self.black_out_non_deck_fn(img, lbl, deck)



        
        return img, lbl

    def deck_intersection(decks):
        deck = np.ones(decks[0].shape).astype(dtype=np.uint32)
        for d in decks:
            deck = deck * d
        return deck
            
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
        # assert len(deck.shape) == 2
        # assert lbl.shape == deck.shape
        deck = deck/255  #so that deck is 1 or 0
        lbl[deck==0] = -1  #make everything none deck as class -1
        deck = np.repeat(deck[:,:,np.newaxis], 3, axis=2)  #duplicate deck into the 3rd dimension
        img = img * deck.astype('uint8') 
        return img, lbl

    def mask_preprocess(self, masks):

        new_mask = -1 * np.ones((masks[0].shape[0], masks[0].shape[1], len(self.class_names)), dtype=np.int32)
        for c in range(len(self.class_names)):
            intersection = np.ones(masks[0].shape)
            union = np.zeros(masks[0].shape)
            for mask in masks:
                if c not in mask:
                    intersection = np.zeros(mask.shape)
                    continue
                intersection *= (mask==c).astype(dtype=np.uint32)
                union += ((mask==c).astype(dtype=np.uint32))
                union = (union>0).astype(dtype=np.uint32)
            new_mask[:, :, c][intersection == 1] = c
            new_mask[:, :, c][(1-union) == 1]= (c + len(self.class_names))

        return new_mask

    def augment_image(self, img, lbl):
    
        seq_det = self.seq.to_deterministic() #seq_det is now a fixes sequence, so lbl and deck get treated the same as img
                                         #it will be a new one after each call of seq.to_deterministic()  
        img = seq_det.augment_image(img)
        lbl = seq_det.augment_image(lbl, hooks=self.hooks_lbl)

        return img, lbl

    def preprocess_fn(self, img):

        img = self.dynamic_histogram_spreading(img)

        return img

    def dynamic_histogram_spreading(self, img, clipLim = 1.0, gridSize = 32):   
        #convert PIL image to cv image in lab colorspace
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        #from stackoverflow https://stackoverflow.com/questions/24341114/simple-illumination-correction-in-images-opencv-c/39744436#39744436
        #-----Converting image to LAB Color model----------------------------------- 
        #lab = cv2.cvtColor(cvimg, cv2.COLOR_BGR2LAB)
        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)
        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=clipLim, tileGridSize=(gridSize, gridSize))
        cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))
        #-----Converting image from LAB Color model to RGB model--------------------
        cvfinal = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)       
        return cvfinal

        

class BridgeSeg(BridgeSegBase):
    def __init__(self, split='train', rcrop=[None, None], use_data_augmentation=False, black_out_non_deck=False, use_class_weight=False, preprocess=False):

       super(BridgeSeg, self).__init__(split=split, use_data_augmentation=use_data_augmentation, black_out_non_deck=black_out_non_deck, use_class_weight=use_class_weight, preprocess=preprocess) 
       if len(rcrop) == 2:
           self.rcrop = np.array(rcrop)
       else:
           self.rcrop = np.array([rcrop, rcrop])


    @staticmethod
    def download():
        print("Not implemented yet (could be a good idea)")
