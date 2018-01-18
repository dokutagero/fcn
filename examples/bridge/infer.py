#!/usr/bin/env python

import pandas as pd
import argparse
import os
import os.path as osp
import re

import chainer
import numpy as np
import skimage.io

import fcn
from fcn.utils import label_accuracy_score


def infer():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('-m', '--model-file')
    parser.add_argument('-i', '--img-files', nargs='+', required=True)
    parser.add_argument('-o', '--out-dir', required=True)
    parser.add_argument('-nc', '--n-class', required=True, type=int)
    args = parser.parse_args()

    # model

    if args.model_file is None:
        args.model_file = fcn.models.FCN8s.download()

    match = re.match('^FCN(32|16|8)s.*$', osp.basename(args.model_file))
    if match is None:
        print('Unsupported model filename: %s' % args.model_file)
        quit(1)
    n_class = args.n_class
    model_name = 'FCN%ss' % match.groups()[0]
    model_class = getattr(fcn.models, model_name)
    model = model_class(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # inference

    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    iu_per_img = np.zeros((1,3))
    if n_class == 4:
        label_names = np.append(fcn.datasets.BridgeSeg.class_names, ['non-deck'])
    else:
        label_names = fcn.datasets.BridgeSeg.class_names

    
    df_list = []
    header = ['img_name', 'damage_level']
    header.extend(label_names)
    for file in args.img_files:
        print(file)
        # input
        img = skimage.io.imread(file, img_num=0)
        mask_path = '/root/teera/bridge_masks/{}/'.format(file.split('/')[-2])
        mask_name = file.split('/')[-1].split('.')[0] + '.png'
        mask = skimage.io.imread(mask_path + mask_name)
        # mask = mask / 255
        mask = color_class_label(mask)
        input, = fcn.datasets.transform_lsvrc2012_vgg16((img,))
        input = input[np.newaxis, :, :, :]
        if args.gpu >= 0:
            input = chainer.cuda.to_gpu(input)

        # forward
        with chainer.no_backprop_mode():
            input = chainer.Variable(input)
            with chainer.using_config('train', False):
                model(input)
                lbl_pred = chainer.functions.argmax(model.score, axis=1)[0]
                lbl_pred = chainer.cuda.to_cpu(lbl_pred.data)


        # visualize
        viz = fcn.utils.visualize_segmentation(
            lbl_true=mask, lbl_pred=lbl_pred, img=file, n_class=n_class,
            label_names=label_names)

        damage_level = file.split('/')[-2]

        img_name = file.split('/')[-1]
        # Compute per image scores
        acc = label_accuracy_score(mask, lbl_pred, n_class)
        # csv_df[label_names] = acc[4] 
        row = [img_name, damage_level]
        row.extend(acc[4].tolist())
        df_list.append(row)
        
        out_file = osp.join(args.out_dir, osp.basename(file))
        skimage.io.imsave(out_file, viz)
        print('==> wrote to: %s' % out_file)


    csv_df = pd.DataFrame.from_records(data=df_list, columns=header)
    csv_df.to_csv('infer.csv', index_label='id')


def color_class_label(image):
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


if __name__ == '__main__':
    infer()
