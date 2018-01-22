#!/usr/bin/env python
import pdb

import argparse
import datetime
import os.path as osp
import subprocess

import chainer
from chainer import cuda

# import fcn
# from fcn import datasets
from fcn import datasets
import fcn.models

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU id')
    parser.add_argument('-da', '--data-augmentation', type=int, \
                        default=0, choices=(0,1),
                        help='Data augmentation flag. Default 0, 1 for data augmentation')
    parser.add_argument('-d', '--deck-mask', type=int, default=1, choices=(0,1),\
                        help='Applying deck mask. Default 1, 0 for not masking deck')
    parser.add_argument('-e', '--epochs', type=int, default=100, choices=range(1000), \
                        help='Number of epochs', metavar='range(0...1000)')
    args = parser.parse_args()


    gpu = args.gpu

    # 0. config

    cmd = 'git log -n1 --format="%h"'
    vcs_version = subprocess.check_output(cmd, shell=True).strip()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = 'fcn32s_VCS-%s_TIME-%s' % (
        vcs_version,
        timestamp,
    )
    out = osp.join(here, 'logs', out)

    # 1. dataset
    deck_flag = bool(args.deck_mask) 
    data_augmentation = bool(args.data_augmentation)
    class_weight_flag = False
    train_dataset = datasets.BridgeSeg(
        split='train',
        rcrop=[512,512],
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=data_augmentation
    )
    train_dataset_nocrop = datasets.BridgeSeg(
        split='train',
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=False
    )
    test_dataset = datasets.BridgeSeg(
        split='validation',
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=False
    )


    iter_train = chainer.iterators.MultiprocessIterator(
        train_dataset, batch_size=1, shared_mem=10 ** 8)
    iter_train_nocrop = chainer.iterators.MultiprocessIterator(
        train_dataset_nocrop, batch_size=1, shared_mem=10 ** 8,
        repeat=False, shuffle=False)
    iter_valid = chainer.iterators.MultiprocessIterator(
        test_dataset, batch_size=1, shared_mem=10 ** 8,
        repeat=False, shuffle=False)

    train_samples = len(train_dataset)
    nbepochs = args.epochs

    # 2. model

    n_class = len(train_dataset.class_names)
    class_weight = train_dataset.class_weight


    model = fcn.models.ResNet101LayersFCN32(n_class=n_class, class_weight=class_weight)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=1.0e-10, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    model.upscore.disable_update()

  # training loop

    # pdb.set_trace()
    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_train_noncrop=iter_train_nocrop,
        iter_valid=iter_valid,
        out=out,
        max_iter=train_samples*nbepochs,
        interval_validate=train_samples
    )
    trainer.train(fold=0)


if __name__ == '__main__':
    main()
