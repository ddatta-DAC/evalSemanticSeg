import evalSemanticSeg
import sys
sys.path.append('../..')
from evalSemanticSeg import cityscapes_label_data
from evalSemanticSeg import eval
from evalSemanticSeg.cityscapes_labels import labelIdConverter
print(cityscapes_label_data.LABEL_DATA_name2id)
import numpy as np

arr1 = np.array([
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 2, 2, 2, 2, 1, 1],
    [1, 1, 1, 2, 1, 1, 2, 3],
    [1, 1, 1, 1, 1, 1, 3, 3],
],dtype=int)

arr2 = np.array([
    [1, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 2, 2, 2, 2, 1, 1],
    [1, 1, 1, 2, 1, 1, 2, 3],
    [1, 1, 1, 2, 2, 1, 2, 3],
],dtype=int)

print(arr1)
num_classes = len(cityscapes_label_data.LABEL_DATA_name2id)
print(labelIdConverter._convert_ndarr_(arr1))
print(eval.mean_iou(arr1,arr2, num_classes= 3, ignore_index=None, nan_to_num=0))
