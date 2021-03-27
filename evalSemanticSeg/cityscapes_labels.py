# --------------------------------------
# Cityscapes dataset has its own labels
# We need to convert to our label ids so that evaluation be performed
# --------------------------------------
import sys
sys.path.append('./..')
import numpy as np
import cityscapes_label_data

csLabelData_name2id = cityscapes_label_data.LABEL_DATA_name2id

class labelIdConverter():
    cityscape_label_ids = list(csLabelData_name2id.values())
    def __init__(self):
        return
    @staticmethod
    def _convert_(input_label_id: int):
        # Currently do an Random transform
        return labelIdConverter.cityscape_label_ids[input_label_id]+2


    @staticmethod
    def _convert_ndarr_(label_arr):
        assert len(label_arr.shape)>=2
        vfunc = np.vectorize(labelIdConverter._convert_)
        return vfunc(label_arr)
        
