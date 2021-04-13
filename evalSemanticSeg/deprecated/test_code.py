
import os
print(os.getcwd())
obj = anotationGen('./../../labelmap.txt', model_output_sparse=False)
file_name = '1008_SS_D_4c4a631994b9e7e02f6de22d28d2b66f7ec9acec3f66cf1b4bb9f36b2387b926ebf9472f06798a77ca3e3d54fd071de6027406deb452036bea8b58d3a7153567_40'
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