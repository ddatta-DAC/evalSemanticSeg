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
            model_output_sparse = True
    ):
        self.num_classes = 0
        if model_output_sparse:
            self.__build_type1_(
                labelmap_file,
                label_col,
                color_col,
                file_col_sep
            )
        else:
            self.__build_type2_(
                labelmap_file,
                label_col,
                color_col,
                file_col_sep
            )
        print(self.color_to_synID)
        global CS_label_names

        self.synID_to_desc ={
            k: CS_label_names[v-1] for k,v in self.synID_2_csID.items()
        }
        return

    def __build_type1_(self,
            labelmap_file,
            label_col,
            color_col,
            file_col_sep):
        # List of class names in CS dataset (ordered)
        global CS_label_names

        # Dict of CS labels such { 1: road, 2: ... , 19: ...}
        cont_CS_labels_id2name = {}
        cont_CS_labels_name2id = {}
        for e in enumerate(CS_label_names, 1):
            cont_CS_labels_id2name[e[0]] = e[1]
            cont_CS_labels_name2id[e[1]] = e[0]
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

        # -------------------
        # Find the set of valid cityscape labels : which are present in cvat
        # -------------------
        valid_CS_labels = set(
            color_to_inputLabel.values()
        ).intersection(set(CS_label_names))
        valid_CS_labels = [_ for _ in CS_label_names if _ in valid_CS_labels]
        # ----------------------------------------------------------
        # The images should contain label ids present in this set
        # ----------------------------------------------------------
        self.valid_CS_label2ID = defaultdict(lambda: None)

        for _ in valid_CS_labels:
            self.valid_CS_label2ID[_] = csLabelData_name2id[_]
        self.valid_CS_label2ID = sorted(self.valid_CS_label2ID.items(), key=itemgetter(1))
        self.valid_CS_label2ID = defaultdict(
            lambda: None,
            {i[0]: i[1] for i in self.valid_CS_label2ID}
        )
        # ---------------------------------------------
        # Find the ids of the CS labels ( e.g: road:7)
        color_to_CSLabelID = OrderedDict({})
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
        for item in valid_CS_labels:
            _ = self.valid_CS_label2ID[item]
            if _ is not None:
                self.synID_2_csID[i] = _
                i += 1

        self.csID_2_synID = {v: k for k, v in self.synID_2_csID.items()}
        self.csID_2_synID[0] = 0

        self.color_to_synID = {
            k: self.csID_2_synID[v] for k, v in self.color_to_CSLabelID.items()}
        return

    def __build_type2_(
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
    # [ Input is the output of SS Model ]
    # -----------------------------------------
    def gen_SynLabel(self, data_path=None):
        print('generateSynLabel', data_path)
        data = np.load(data_path)

        def _replace(val):
            if val in self.csID_2_synID.keys():
                return self.csID_2_synID[val]
            else:
                return 0
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


# -------------------------------------
'''
# Sample code
obj = anotationGen('./../labelmap.txt')
res = obj.generateContLabel('./../4.jpg.npy')
res = obj.processSegMask('./../tmp.png')
'''

# import os
# print(os.getcwd())
# obj = anotationGen('./../../labelmap.txt', model_output_sparse=False)
#
# ss_op_path = './../../Data/seg_results/1464_SS_D_2efde18d0e007c1512bc73edc187170e0b1550af8cd8fe6fd4dedd6136811e6bc90cb54591ac705afbf9cbeda7109cff85ccbb9c1a240358325c010752df71e2_36.npy'
# print(obj.csID_2_synID)
# ss = obj.gen_SynLabel( data_path=ss_op_path)
# print(ss.shape)
#
# print(obj.synID_to_desc)
# print('generateSynLabel', ss[300,550:575])
#
#
# ss_mask_path = './../../Data/img/1472_SS_D_81b60ba6e44150ac736bc7f21182d49c094afeb0b6b06aba474a3114458dad3972eba78147d60b9b799784e70f0dd41d52509748c62b3b588f8bda50f7d8d163_21.png'
# img = cv2.cvtColor(cv2.imread(ss_mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
# print('>',img[300,550:575])
# res = obj.process_SegMask(ss_mask_path)
# print(res[300,550:575])

