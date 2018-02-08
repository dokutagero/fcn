from .eval_ious import calc_semantic_segmentation_confusion
from .eval_ious import calc_semantic_segmentation_iou
import os.path as osp 
import fcn
import chainer
from PIL import Image
from scripts import label2mask as l2m
import numpy as np


DATASET_BRIDGE_DIR = osp.expanduser('/home/dokutagero/repos/dataset_bridge/')

def labels_from_xml(files, dataset_train_nocrop):
    lbls = []
    imgs = []
    for f in files:
        img = Image.open(f['img'])
        lbl_files = []
        lbls_img = []
        for label_file in f['lbl']:
            print(label_file)
            imsize = img.size
            lbl_files.append(osp.join(DATASET_BRIDGE_DIR, 'bridge_masks_xml/', label_file))
        for m in lbl_files:
            lbls_img.append(dataset_train_nocrop.color_class_label(l2m(m, imsize)))

        lbls.append(lbls_img) 
        imgs.append(f['img'])

    return imgs, lbls
    # lbl = l2m(lbl_name, imsize)

dataset_train_nocrop = fcn.datasets.BridgeSeg(
    split='train_xml_uniq',
    use_class_weight=False,
    black_out_non_deck=True,
    use_data_augmentation=False
)

imgs, lbls = labels_from_xml(dataset_train_nocrop.files['train_xml'], dataset_train_nocrop)

# build the pairs to test
ranking_iou_lbls = []
imgs_filtered = []
for idx, img_lbls in enumerate(lbls):
    print(idx)
    if len(img_lbls)==3:
        ranking_iou_lbls.append([(img_lbls[0], img_lbls[0], img_lbls[1]), (img_lbls[1], img_lbls[2], img_lbls[2])])
        imgs_filtered.append(imgs[idx])



confusions = []
ious = []
iou_means = []
for elems in ranking_iou_lbls:
    confusions.append(calc_semantic_segmentation_confusion(elems[0], elems[1]))
    ious.append(calc_semantic_segmentation_iou(confusions[-1]))
    iou_means.append(np.mean(ious[-1][1:]))
