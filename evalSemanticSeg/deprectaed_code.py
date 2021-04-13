import numpy as np
from pprint import pprint

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


# ------------

import os
print(os.getcwd())
obj = anotationGen('./../../labelmap.txt', model_output_sparse=False)
file_name = '1010_SS_D_89f77e6ed4f3a8b4b9199e68b130cdd83a2159f5ece4041ca72aa49164b0c8bb5b7b5732754a9a916fa766ccc20b14d5ee04771fde8739002e6195380814be4a_42'
model_op_path = './../../Data/seg_results/{}.npy'.format(file_name)
pprint(obj.csID_2_synID)
prediction = obj.gen_SynLabel( data_path=model_op_path)

# print(prediction.shape)

print(obj.synID_to_desc)
print('generateSynLabel', prediction[300,550:575])
ss_mask_path = './../../Data/img/{}.png'.format(file_name)

img = cv2.cvtColor(cv2.imread(ss_mask_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.show()
# print('>',img[300,550:575])
ground_truth = obj.process_SegMask(ss_mask_path)
# print(ground_truth[300,550:575])

valid_class_labels = list(obj.synID_to_desc.keys())
for _class_label_ in valid_class_labels:
    mask = np.ones(ground_truth.shape, dtype=int)
    mask = mask * int(_class_label_)
    gt = np.equal(ground_truth, mask).astype(int)
    pred = np.equal(prediction, mask).astype(int)

    _intersection = np.logical_and(gt,pred)
    _union = np.logical_or(gt,pred)

    if np.sum(_union)>0:
        IoU = np.sum(_intersection)/np.sum(_union)
    else:
        IoU = 0

    print(_class_label_,IoU)