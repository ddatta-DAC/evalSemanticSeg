import os
import sys
import numpy as np

# ------------------------------
import pandas as pd

sys.path.append('.')
'''
Obtain valid_class_labels from annoattor object
'''


class metrics_object:
    def __init__(self, valid_class_labels):
        self.valid_class_labels = valid_class_labels
        return

    # -------------------------------
    # Ensure that the prediction has values 1...n for valid labels
    # 0 is meant for unknown
    # -------------------------------
    def evaluate(
            self,
            ground_truth,
            prediction
    ):
        result_dict = {}
        for _class_label_ in self.valid_class_labels:
            mask = np.ones(ground_truth.shape, dtype=int)
            mask = mask * int(_class_label_)
            gt = np.equal(ground_truth, mask).astype(int)
            pred = np.equal(prediction, mask).astype(int)

            _intersection = np.logical_and(gt, pred)
            _union = np.logical_or(gt, pred)

            if np.sum(_union) > 0:
                IoU = np.sum(_intersection) / np.sum(_union)
            else:
                IoU = 0
            result_dict[_class_label_] = IoU
        return result_dict

    '''
    Helper function to assimilate the results of multiple images
    '''

    def collate(self, list_result_dict, synID_to_desc=None):
        results = {c: [] for c in self.valid_class_labels}
        for _dict in list_result_dict:
            for c, v in _dict.items():
                results[c].append(v)

        for c in self.valid_class_labels:
            results[c] = np.mean(results[c])
        if synID_to_desc is not None:
            labelled_results = {}
            for c in self.valid_class_labels:
                labelled_results[synID_to_desc[c]] = results[c]
            return labelled_results
        else:
            return results
