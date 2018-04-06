from __future__ import division

import numpy as np
import six
import pdb


def calc_semantic_segmentation_confusion_uncert(pred_labels, gt_labels):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    channel_max = 0
    confusion = np.zeros((n_class, n_class*2), dtype=np.int64)
    iou_dict = {0 : [0, 0], 1 : [0, 0], 2 : [0, 0]}
    hits = np.zeros((1,3))
    errors = np.zeros((1,3))
    # mc = multichannel
    for pred_label, gt_label_mc in six.moves.zip(pred_labels, gt_labels):
        for channel in range(gt_label_mc.shape[-1]):
            # pdb.set_trace()
            if pred_label.ndim != 2 or gt_label_mc[:,:,channel].ndim != 2:
                raise ValueError('ndim of labels should be two.')
            if pred_label.shape != gt_label_mc[:,:,channel].shape:
                raise ValueError('Shape of ground truth and prediction should'
                                 ' be same.')
            # pdb.set_trace()
            pred_label_flat = pred_label.flatten()
            gt_label = gt_label_mc[:,:,channel].flatten()

            # Dynamically expand the confusion matrix if necessary.
            # lb_max = max((pred_label_flat +  gt_label))
            # if lb_max >= n_class:
            #     expanded_confusion = np.zeros(
            #         ((lb_max + 1), (lb_max + 1)), dtype=np.int64)
            #     expanded_confusion[0:n_class, 0:n_class] = confusion

            #     n_class = lb_max + 1
            #     confusion = expanded_confusion

            # Count statistics from valid pixels.
            mask = gt_label >= 0
            res = pred_label_flat[mask] + gt_label[mask]
            # if channel == 0:
            #     hits = len(np.where(res == 0)[0])
            #     errors = len(np.where(res == 3)[0])
            # if channel == 1:
            #     hits = len(np.where(res == 2)[0])
            #     errors = len(np.where(res == 5)[0])
            # if channel == 2:
            #     hits = len(np.where(res == 4)[0])
            #     errors = len(np.where(res == 7)[0])
            hits[0,channel] += (res == channel*2).sum()
            errors[0,channel] += (res == (2 * channel + 3)).sum()
            
            # iou_dict[channel][0] = iou_dict[channel][0] + hits
            # iou_dict[channel][1] = iou_dict[channel][1] + errors
            # confusion += np.bincount(
            #     (n_class) * gt_label[mask].astype(int) +
            #     pred_label_flat[mask], minlength=(channel_max * lb_max)).reshape((channel_max, lb_max))
            #import pdb; pdb.set_trace()
            

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return hits, errors


def calc_semantic_segmentation_iou_uncert(hits, errors):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    # nclass = confusion.shape[0]//2
    # # pdb.set_trace()
    # iou_numerator = np.diag(confusion[:nclass, :nclass])
    # iou_denominator = np.diag(confusion[:nclass,nclass:])
    # # iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0)
    # #                    - np.diag(confusion))
    # iou = iou_numerator / iou_denominator
    # iou_num = np.array([iou_dict[0][0], iou_dict[1][0], iou_dict[2][0]])
    # iou_den = np.array([iou_dict[0][1], iou_dict[1][1], iou_dict[2][1]])
    # iou = iou_num / (iou_num + iou_den)
    iou = hits / (hits + errors)
    return iou


def eval_semantic_segmentation_uncert(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`

    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): A collection of predicted
            labels. The shape of a label array
            is :math:`(H, W)`. :math:`H` and :math:`W`
            are height and width of the label.
            For example, this is a list of labels
            :obj:`[label_0, label_1, ...]`, where
            :obj:`label_i.shape = (H_i, W_i)`.
        gt_labels (iterable of numpy.ndarray): A collection of ground
            truth labels. The shape of a ground truth label array is
            :math:`(H, W)`, and its corresponding prediction label should
            have the same shape.
            A pixel with value :obj:`-1` will be ignored during evaluation.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    # confusion = calc_semantic_segmentation_confusion_uncert(
    #     pred_labels, gt_labels)
    hits, errors = calc_semantic_segmentation_confusion_uncert(
         pred_labels, gt_labels)
    # iou = calc_semantic_segmentation_iou_uncert(confusion)
    iou = calc_semantic_segmentation_iou_uncert(hits, errors)
    # pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    # class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {'iou': iou, 'miou': np.mean(iou), 'miou_damage' : np.mean(iou[1:])}
            # 'pixel_accuracy': pixel_accuracy,
            # 'class_accuracy': class_accuracy,
            # 'mean_class_accuracy': np.nanmean(class_accuracy)}
