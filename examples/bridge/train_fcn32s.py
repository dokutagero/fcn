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

    # pdb.set_trace()
    dataset_train = datasets.BridgeSeg(split='train', rcrop=[400,400])
    dataset_train_nocrop = datasets.BridgeSeg(split='train')
    dataset_valid = datasets.BridgeSeg(split='validation')

    iter_train = chainer.iterators.MultiprocessIterator(
        dataset_train, batch_size=1, shared_mem=10 ** 7)
    iter_train_nocrop = chainer.iterators.MultiprocessIterator(
        dataset_train_nocrop, batch_size=1, shared_mem=10 ** 7)
    iter_valid = chainer.iterators.MultiprocessIterator(
        dataset_valid, batch_size=1, shared_mem=10 ** 7,
        repeat=False, shuffle=False)

    train_samples = len(dataset_train)
    print(train_samples)
    nbepochs = 100
  # 2. model

    n_class = len(dataset_train.class_names)

    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg.pretrained_model, vgg)

    model = fcn.models.FCN32s(n_class=n_class)
    model.init_from_vgg16(vgg)

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
        iter_train_nocrop=iter_train_nocrop,
        iter_valid=iter_valid,
        out=out,
        max_iter=train_samples*nbepochs,
    )
    trainer.train()


if __name__ == '__main__':
    main()
