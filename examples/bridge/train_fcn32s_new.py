#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp

os.environ['MPLBACKEND'] = 'Agg'  # NOQA

import chainer
from chainer.training import extensions
import chainercv
import fcn


here = osp.dirname(osp.abspath(__file__))


def get_data(deck_flag, data_augmentation):
    dataset_train = fcn.datasets.BridgeSeg(split='train')

    class_names = dataset_train.class_names

    # Include this parameters in main function
    deck_flag = bool(deck_flag) 
    data_augmentation = bool(data_augmentation)
    class_weight_flag = False



    dataset_train = datasets.BridgeSeg(
        split='train',
        rcrop=[512,512],
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=data_augmentation
    )

    dataset_train_nocrop = datasets.BridgeSeg(
        split='train',
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=False
    )

    dataset_valid = datasets.BridgeSeg(
        split='validation',
        use_class_weight=class_weight_flag,
        black_out_non_deck=deck_flag,
        use_data_augmentation=False
    )

    # Apply per channel mean substraction
    dataset_train = chainer.datasets.TransformDataset(
        dataset_train, fcn.datasets.transform_lsvrc2012_vgg16)
    dataset_valid = chainer.datasets.TransformDataset(
        dataset_valid, fcn.datasets.transform_lsvrc2012_vgg16)
    dataset_train_nocrop = chainer.datasets.TransformDataset(
        dataset_train_nocrop, fcn.datasets.transform_lsvrc2012_vgg16)
    num_train_samples = len(dataset_train)
    
    # Create iterators
    iter_train = chainer.iterators.SerialIterator(
        dataset_train, batch_size=1)
    iter_valid = chainer.iterators.SerialIterator(
        dataset_valid, batch_size=1, repeat=False, shuffle=False)
    iter_train_nocrop = chainer.iterators.SerialIterator(
        dataset_train_nocrop, batch_size=1, repeat=False, shuffle=False)

    return num_train_samples, class_names, iter_train, iter_valid, iter_train_nocrop


def get_trainer(optimizer, iter_train, iter_valid, iter_train_nocrop,
                class_names, args):
    model = optimizer.target

    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=args.gpu)

    trainer = chainer.training.Trainer(
        updater, (args.max_iteration, 'iteration'), out=args.out)

    trainer.extend(fcn.extensions.ParamsReport(args.__dict__))

    trainer.extend(extensions.ProgressBar(update_interval=5))

    trainer.extend(extensions.LogReport(
        trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'elapsed_time',
         'main/loss', 'validation/main/miou']))

    def pred_func(x):
        model(x)
        return model.score

    # trainer.extend(
    #     fcn.extensions.SemanticSegmentationVisReport(
    #         pred_func, iter_valid_raw,
    #         transform=fcn.datasets.transform_lsvrc2012_vgg16,
    #         class_names=class_names, device=args.gpu, shape=(4, 2)),
    #     trigger=(args.interval_eval, 'iteration'))

    trainer.extend(
        chainercv.extensions.SemanticSegmentationEvaluator(
            iter_valid, model, label_names=class_names),
        trigger=(args.interval_eval, 'iteration'))

    trainer.extend(
        chainercv.extensions.SemanticSegmentationEvaluator(
            iter_train_nocrop, model, label_names=class_names),
        trigger=(args.interval_eval, 'iteration'))

    trainer.extend(extensions.snapshot_object(
        target=model, filename='model_best.npz'),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/miou',
            trigger=(args.interval_eval, 'iteration')))

    assert extensions.PlotReport.available()
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss'], x_key='iteration',
        file_name='loss.png', trigger=(args.interval_print, 'iteration')))
    trainer.extend(extensions.PlotReport(
        y_keys=['validation/main/miou'], x_key='iteration',
        file_name='miou.png', trigger=(args.interval_print, 'iteration')))

    return trainer


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

    args.model = 'FCN32s'
    args.lr = 1e-10
    args.momentum = 0.99
    args.weight_decay = 0.0005

    args.interval_print = 20

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S'))

    # data
    num_train_samples, class_names, iter_train, iter_valid, iter_train_nocrop = get_data(args.deck_mask, \
                                                                      args.data_augmentation)
    n_class = len(class_names)
    args.max_iteration = args.epochs * train_samples
    args.interval_eval = train_samples

    # model
    vgg = fcn.models.VGG16()
    chainer.serializers.load_npz(vgg.pretrained_model, vgg)
    model = fcn.models.FCN32s(n_class=n_class)
    model.init_from_vgg16(vgg)

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
    model.upscore.disable_update()

    # trainer
    trainer = get_trainer(optimizer, iter_train, iter_valid, iter_train_nocrop,
                          class_names, args)
    trainer.run()


if __name__ == '__main__':
    main()
