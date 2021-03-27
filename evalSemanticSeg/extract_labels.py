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
from collections import OrderedDict, defaultdict
import cityscapes_label_data
import cvat_to_cityscape_labels

csLabelData_name2id = cityscapes_label_data.LABEL_DATA_name2id
valid_CS_classLabels = cityscapes_label_data.target_classLabels

csLabelData_id2Name = defaultdict(lambda: -1)
for k, v in csLabelData_name2id.items():
    csLabelData_id2Name[v] = k
cvat_2_cityscapes = cvat_to_cityscape_labels.cvat_2_cityscapes


# ========================================================
class anotationGen:
    '''
    Input file should
    '''

    def __init__(
            self,
            labelmap_file,
            label_col='# label',
            color_col='color_rgb',
            file_col_sep=':'
    ):
        global valid_CS_classLabels
        _df_ = pd.read_csv(
            labelmap_file,
            sep=file_col_sep,
            index_col=None
        )
        default_label = 'none'
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
        # Find the set of valid cityscape labels : which are present in cvat
        valid_CS_labels = set(color_to_inputLabel.values()).intersection(set(valid_CS_classLabels))
        # ----------------------------------------------------------
        # The images should contain label ids present in this set
        # ----------------------------------------------------------
        self.valid_CS_label2ID = defaultdict(lambda: None)
        for _ in valid_CS_labels:
            self.valid_CS_label2ID[_] = csLabelData_name2id[_]
        print(self.valid_CS_label2ID)

        # Find the ids of the CS labels ( e.g: road:7)
        color_to_CSLabelID = {}
        for k, v in color_to_inputLabel.items():
            if v in valid_CS_labels:
                color_to_CSLabelID[k] = csLabelData_name2id[v]
            else:
                color_to_CSLabelID[k] = 0
        self.color_to_CSLabelID = color_to_CSLabelID
        # Account for the "background" class
        self.num_classes = len(self.valid_CS_label2ID) + 1

        self.synID_2_csID = {}
        self.csID_2_synID = {}
        # -----------------------------------------------
        # Calculate mapping from CS labels to  0 ... n
        # 1: 7(road), 2: 8(sidewalk)
        # -----------------------------------------------
        i = 1
        for item in valid_CS_classLabels:
            _ = self.valid_CS_label2ID[item]
            if _ is not None:
                self.synID_2_csID[i] = _
                i += 1
        print(self.synID_2_csID)
        self.csID_2_synID = {v: k for k, v in self.synID_2_csID.items()}
        return

    # -----------------------------------------
    # Take output of Semantic Segmentation and convert it to continuous label ids
    # -----------------------------------------
    def generateContLabel(self, data_path=None):

        data = np.load(data_path)
        print(np.unique(data))

        def _replace(val):
            if val in self.csID_2_synID.keys():
                return self.csID_2_synID[val]
            else:
                return 0

        vfunc = np.vectorize(_replace)
        processedLabels = vfunc(data)
        return processedLabels

    # -----------------------------------------
    # Read in the segmented image, and generate continuous ids
    # -----------------------------------------
    def processSegMask(self, img_path=None):
        image = cv2.imread(
            img_path,
            cv2.IMREAD_COLOR)

        def _replace(val):
            _key = (val[0], val[1], val[2])
            if _key in self.color_to_inputLabel.keys():
                _name = self.color_to_inputLabel[_key]
                if _name == 'none':
                    return 0
                cslabelID = self.valid_CS_label2ID[_name]
                contlabelID = self.csID_2_synID[cslabelID]
                return contlabelID
            return 0

        res = np.apply_along_axis(_replace, 2, image)
        return res


# -------------------------------------
'''
# Sample code
obj = anotationGen('./../untitled.txt')
res = obj.generateContLabel('./../4.jpg.npy')
res = obj.processSegMask('./../tmp.png')
'''
