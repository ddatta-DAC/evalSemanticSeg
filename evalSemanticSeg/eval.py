import os
import sys
import numpy as np

# ------------------------------
sys.path.append('.')

'''
Adapted from mmsegmentation library
'''


def mean_iou(
        predictions,
        gt_seg_maps,
        num_classes,
        ignore_index,
        nan_to_num=None,
        reduce_zero_label=False
):
    all_acc, acc, iou = eval_metrics(
        predictions=predictions,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, iou


def mean_dice(
        predictions,
        gt_seg_maps,
        num_classes,
        ignore_index,
        nan_to_num=None,
        reduce_zero_label=False
):
    all_acc, acc, dice = eval_metrics(
        predictions=predictions,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        reduce_zero_label=reduce_zero_label)
    return all_acc, acc, dice


'''
This works for a single image
'''


def eval_metrics(
        predictions,
        gt_seg_maps,
        num_classes,
        ignore_index,
        metrics=['mIoU'],
        nan_to_num=None,
        reduce_zero_label=False
):
    if type(predictions) == list:
        predictions = np.array(predictions)

    if type(gt_seg_maps) == list:
        gt_seg_maps = np.array(gt_seg_maps)

    if len(predictions.shape) == 2: predictions = np.expand_dims(predictions, 0)
    if len(gt_seg_maps.shape) == 2: gt_seg_maps = np.expand_dims(gt_seg_maps, 0)

    if isinstance(metrics, str):
        metrics = [metrics]

    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        predictions,
        gt_seg_maps,
        num_classes,
        ignore_index,
        reduce_zero_label
    )
    sum_total_area_label = np.sum(total_area_label)
    all_acc = total_area_intersect.sum() / sum_total_area_label
    acc = total_area_intersect / total_area_label

    ret_metrics = [all_acc, acc]
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


def total_intersect_and_union(predictions,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              reduce_zero_label=False):
    num_imgs = predictions.shape[0]
    assert gt_seg_maps.shape[0] == num_imgs
    total_area_intersect = np.zeros((num_classes,), dtype=np.float)
    total_area_union = np.zeros((num_classes,), dtype=np.float)
    total_area_pred_label = np.zeros((num_classes,), dtype=np.float)
    total_area_label = np.zeros((num_classes,), dtype=np.float)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            predictions[i],
            gt_seg_maps[i],
            num_classes,
            ignore_index,
            reduce_zero_label
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def intersect_and_union(
        pred_label,
        label,
        num_classes,
        ignore_index,
        reduce_zero_label=False
):
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]
    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred_label, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label
