'''
This code reads image and creates annotations
Annotation format:
Numpy 2-D array
Values : Class label ids : Continuous values 0-18  - Since cityscapes models have 19 classes
'''
from pprint import pprint
import pandas as pd
import numpy as np
import cv2
import os
from pprint import pprint
import sys
from operator import itemgetter
from collections import OrderedDict, defaultdict
try:
    from . import cityscapes_label_data
    from . import cvat_to_cityscape_labels
except:
    import cityscapes_label_data
    import cvat_to_cityscape_labels

csLabelData_name2id = cityscapes_label_data.LABEL_DATA_name2id
CS_label_names = cityscapes_label_data.target_classLabels

csLabelData_id2Name = defaultdict(lambda: -1)
for k, v in csLabelData_name2id.items():
    csLabelData_id2Name[v] = k
cvat_2_cityscapes = cvat_to_cityscape_labels.cvat_2_cityscapes


# ========================================================
# Use model_output_sparse =True if input labels are 7,8,...33
# Use model_output_sparse =True if input labels are 1,2,...19
#=========================================================
class anotationGen:
    '''
    Input file should
    '''
    def __init__(
            self,
            labelmap_file,
            label_col='# label',
            color_col='color_rgb',
            file_col_sep=':',
            model_output_sparse = False,
            exclude_cvat_classes = ['hood']
    ):
        global CS_label_names
        self.num_classes = 0
        self.exclude_cvat_classes = exclude_cvat_classes
        self.__build_(
            labelmap_file,
            label_col,
            color_col,
            file_col_sep
        )

        # Convert from 7,8 ... 33 to 1....
        if model_output_sparse:
            pass
            # TODO : do simple mapping


        self.synID_to_desc ={
            k: CS_label_names[v-1] for k,v in self.synID_2_csID.items()
        }
        print(self.synID_to_desc)
        pprint(self.color_to_synID)
        return


    def __build_(
            self,
            labelmap_file,
            label_col,
            color_col,
            file_col_sep):
        # List of class names in CS dataset (ordered)
        global CS_label_names

        # Output of CVAT system
        # format rgb: cvat_class_name
        _df_ = pd.read_csv(
            labelmap_file,
            sep=file_col_sep,
            index_col=None
        )

        default_label = None
        # color_to_inputLabel has tupe(rgb) : cityscape label
        color_to_inputLabel = {}

        for i, row in _df_.iterrows():
            input_label = row[label_col]
            _color = row[color_col].strip().split(',')
            _color = (int(_color[0]), int(_color[1]), int(_color[2]))
            if input_label in cvat_2_cityscapes.keys():
                color_to_inputLabel[_color] = cvat_2_cityscapes[input_label]
            else:
                color_to_inputLabel[_color] = default_label

        self.color_to_inputLabel = color_to_inputLabel

        valid_CS_labels = set(
            color_to_inputLabel.values()
        ).intersection(set(CS_label_names))
        valid_CS_labels = [_ for _ in CS_label_names if _ in valid_CS_labels]
        label_name2synID = defaultdict(lambda : None)
        for e in enumerate(valid_CS_labels,1):
            label_name2synID[e[1]] = e[0]
        self.color_to_synID = {}


        for color, lname in self.color_to_inputLabel.items():
            _id = label_name2synID[lname]
            if _id is None:
                _id = 0
            self.color_to_synID[color] = _id
        self.csID_2_synID = defaultdict(lambda : 0)
        i = 1
        for csl in enumerate(CS_label_names,1):
            if csl[1] in valid_CS_labels:
                self.csID_2_synID[csl[0]] = i
                i+=1
        self.synID_2_csID = { v:k for k,v in self.csID_2_synID.items()}
        self.num_classes = len(self.csID_2_synID) + 1
        return


    # -----------------------------------------
    # Take output of Semantic Segmentation and convert it to continuous label ids
    # Input is the output of SS Model ( numpy array object )
    # -----------------------------------------
    def gen_SynLabel(self, data_path=None, default_value = 0 ):
        print('[generating synthetic label]', data_path)
        data = np.load(data_path)
        print(np.min(data), np.max(data), np.unique(data))
        def _replace(val):
            if val in self.csID_2_synID.keys():
                return self.csID_2_synID[val]
            else:
                return default_value

        vfunc = np.vectorize(_replace)
        processedLabels = vfunc(data)
        return processedLabels

    # -----------------------------------------
    # Read in the segmented image, and generate continuous ids [ Input is the ground truth ]
    # -----------------------------------------
    def process_SegMask(self, img_path=None):
        image = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        def _replace(val):
            _key = (val[0], val[1], val[2])
            return self.color_to_synID[_key]

        res = np.apply_along_axis(_replace, 2, image)
        return res

    def get_valid_class_labels(self):
        return list(self.synID_to_desc.keys())

    def get_synID_to_desc(self):
        return self.synID_to_desc
# -------------------------------------





