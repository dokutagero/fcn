#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import subprocess

import chainer
from chainer import cuda

import fcn
from fcn import datasets


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu', type=int, required=True, help='GPU id')
    parser.add_argument(
        '--fcn32s-file', default=fcn.models.FCN32s.pretrained_model,
        help='Pretrained model file of FCN32s')
    args = parser.parse_args()

    gpu = args.gpu
    fcn32s_file = args.fcn32s_file

    # 0. config

    cmd = 'git log -n1 --format="%h"'
    vcs_version = subprocess.check_output(cmd, shell=True).strip()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    out = 'fcn16s_VCS-%s_TIME-%s' % (
        vcs_version,
        timestamp,
    )
    out = osp.join(here, 'logs', out)
    if not osp.exists(out):
        os.makedirs(out)
    with open(osp.join(out, 'config.yaml'), 'w') as f:
        f.write('fcn32s_file: %s\n' % fcn32s_file)

    # 1. dataset

    dataset_train = datasets.BridgeSeg(split='train', rcrop=[400,400])
    dataset_train_nocrop = datasets.BridgeSeg(split='train')
    dataset_valid = datasets.BridgeSeg(split='validation')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=1, shared_mem=10 ** 7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid, batch_size=1, shared_mem=10 ** 7,
        repeat=False, shuffle=False)
    iter_train_nocrop = chainer.iterators.MultiprocessIterator(
        dataset_train_nocrop, batch_size=1, shared_mem=10 ** 7,
        repeat=False, shuffle=False)

    train_samples = len(dataset_train)
    print(train_samples)
    nbepochs = 100

    # 2. model

    n_class = len(dataset_train.class_names)
    class_weight = dataset_train.class_weight

    fcn32s = fcn.models.FCN32s(n_class=n_class, class_weight=class_weight)
    chainer.serializers.load_npz(fcn32s_file, fcn32s)

    model = fcn.models.FCN16s(n_class=n_class, class_weight=class_weight)
    model.init_from_fcn32s(fcn32s)

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()

    # 3. optimizer

    optimizer = chainer.optimizers.MomentumSGD(lr=1.0e-12, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0005))
    for p in model.params():
        if p.name == 'b':
            p.update_rule = chainer.optimizers.momentum_sgd.MomentumSGDRule(
                lr=optimizer.lr * 2, momentum=0)
    model.upscore2.disable_update()
    model.upscore16.disable_update()

    # training loop

    trainer = fcn.Trainer(
        device=gpu,
        model=model,
        optimizer=optimizer,
        iter_train=iter_train,
        iter_train_noncrop=iter_train_nocrop,
        iter_valid=iter_valid,
        out=out,
        max_iter=train_samples*nbepochs,
        interval_validate=train_samples,
    )
    trainer.train()


if __name__ == '__main__':
    main()
