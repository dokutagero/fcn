#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
import fcn

from train_fcn32s_new_trainval import get_data
from train_fcn32s_new_trainval import get_trainer


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='gpu id')
    parser.add_argument(
        '--fcn16s-file', default=fcn.models.FCN16s.pretrained_model,
        help='pretrained model file of FCN16s')
    parser.add_argument('-da', '--data-augmentation', type=int, \
                        default=0, choices=(0,1),
                        help='Data augmentation flag. Default 0, 1 for data augmentation')
    parser.add_argument('-d', '--deck-mask', type=int, default=1, choices=(0,1),\
                        help='Applying deck mask. Default 1, 0 for not masking deck')
    parser.add_argument('-e', '--epochs', type=int, default=100, choices=range(1000), \
                        help='Number of epochs', metavar='range(0...1000)')
    parser.add_argument('-x', '--xval', type=int, default=5)
    parser.add_argument('-t', '--tstrategy', type=int, default=0, choices=(0,1))
    parser.add_argument('-lu', '--learnable', type=int, default=0, choices=(0,1))
    parser.add_argument('-u', '--uncertainty', type=int, default=0, choices=(0,1))
    parser.add_argument('-bs', '--bsize', type=int, default=4)
    args = parser.parse_args()

    args.model = 'FCN8s'
    args.lr = 1e-14
    args.momentum = 0.99
    args.weight_decay = 0.0005

    args.max_iteration = 100000
    args.interval_print = 20
    args.interval_eval = 4000

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()

    # data
    num_train_samples, class_names, dataset_train, dataset_nocrop_train, dataset_nocrop_val, dataset_nocrop_train_uncert, dataset_nocrop_val_uncert = get_data(args.deck_mask, \
                                                                      args.data_augmentation, args.tstrategy, args.uncertainty)

    experiment_name = 'fcn8' + '_uncertainty_' + str(args.uncertainty) + '_da_' + str(args.data_augmentation) + '_ts_' + str(args.tstrategy) + '_lu_' + str(args.learnable)
    args.out = osp.join(here, 'logs', experiment_name + '_' + now.strftime('%Y%m%d_%H%M%S'))
    n_class = len(class_names)

    iter_train = chainer.iterators.MultiprocessIterator(
                 dataset_train, batch_size=args.bsize, n_prefetch=16, n_processes=16, repeat=True, shuffle=True)
    iter_valid = chainer.iterators.MultiprocessIterator(
                 dataset_nocrop_val, batch_size=8, n_prefetch=16, n_processes=16, repeat=False, shuffle=False, shared_mem=100000000)
    iter_train_nocrop = chainer.iterators.MultiprocessIterator(
                 dataset_nocrop_train, batch_size=8, n_prefetch=16, n_processes=16,repeat=False, shuffle=False, shared_mem=100000000)
    iter_train_uncert = chainer.iterators.MultiprocessIterator(
                 dataset_nocrop_train_uncert, batch_size=8, n_prefetch=16, n_processes=16,repeat=False, shuffle=False, shared_mem=100000000)
    iter_valid_uncert = chainer.iterators.MultiprocessIterator(
                 dataset_nocrop_val_uncert, batch_size=8, n_prefetch=16, n_processes=16,repeat=False, shuffle=False, shared_mem=100000000)


    # model
    fcn16s = fcn.models.FCN16s(n_class=n_class)
    chainer.serializers.load_npz(args.fcn16s_file, fcn16s)
    model = fcn.models.FCN8s(n_class=n_class)
    model.init_from_fcn16s(fcn16s)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # optimizer
    optimizer = chainer.optimizers.MomentumSGD(
        lr=args.lr, momentum=args.momentum)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.weight_decay))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    if args.learnable == 0:
        model.upscore2.disable_update()
        model.upscore_pool4.disable_update()
    model.upscore8.disable_update()

    # trainer

    trainer = get_trainer(optimizer, iter_train, iter_valid, iter_train_nocrop, iter_valid_uncert, iter_train_uncert,
                          class_names, args)
    trainer.run()


if __name__ == '__main__':
    main()
