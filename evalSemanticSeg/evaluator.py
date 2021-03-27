import os
import sys
import numpy as np

# ------------------------------
sys.path.append('.')
import tensorflow as tf

class metrics_object:
    def __init__(self,num_classes=19):
        self.metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)
        self.reset()

    def reset(self):
        self.metric.reset_states()

    # ----------------------------------------
    # Single image
    # ----------------------------------------
    def mIoU(self, y_pred, y_true):
        assert y_pred.shape[1] == y_true.shape[1]
        assert y_pred.shape[0] == y_true.shape[0]
        self.metric.update_state(
            y_pred, y_true
        )
        self.reset()
        result = self.metric.result().numpy()
        return result


