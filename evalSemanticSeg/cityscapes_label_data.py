from collections import OrderedDict

'''
Original lables
'''
LABEL_DATA_name2id = OrderedDict({
    'unlabeled': 0,
    'egovehicle': 1,
    'rectificationborder': 2,
    'outofroi': 3,
    'static': 4,
    'dynamic': 5,
    'ground': 6,
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'railtrack': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guardrail': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'polegroup': 18,
    'trafficlight': 19,
    'trafficsign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33,
    'licenseplate': -1
})

target_classLabels = ('road', 'sidewalk',
            'building', 'wall', 'fence', 'pole',
            'trafficlight', 'trafficsign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle'
)
