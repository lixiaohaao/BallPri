#%%
import os
import keras
import numpy as np
from keras.applications import VGG16, VGG19, inception_v3, resnet50, mobilenet
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras_preprocessing import image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import matplotlib.pyplot as plt

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    input_img_data = img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

#%%
ImageNet_dir = '/data0/jinhaibo/lixiaohao/dataset/'
path_list = os.listdir(ImageNet_dir)
path_list.sort(key=lambda x: int(x[-10:-5]))
image_list = []
for image in path_list:
    tmp_image = preprocess_image(ImageNet_dir + image)
    tt = tmp_image.copy()
    image_list.append(tt.astype(int))
image_arrays = np.array(image_list)
image_arrays = image_arrays[:, 0, :, :, :]
ture_label = np.load('/data0/jinhaibo/lixiaohao/dataset_1/label_val.npy')
#%%
resnet_model = resnet50.ResNet50(weights='imagenet')
prediction = resnet_model.predict(image_arrays)
label = np.argmax(prediction, axis=1)
ratio = np.sum(label == ture_label)/len(label)
label_inf = decode_predictions(prediction, top=1)

#%%
correct_class = image_arrays[label == ture_label]
correct_label = ture_label[label == ture_label]
incorrect_class = image_arrays[label != ture_label]
incorrect_label = ture_label[label != ture_label]

#%%
np.save('/data0/jinhaibo/lixiaohao/ImageNet/Benign_exaple.npy', correct_class)
np.save('/data0/jinhaibo/lixiaohao/ImageNet/Benign_label.npy', correct_label)
np.save('/data0/jinhaibo/lixiaohao/ImageNet/FalsePos_exaple.npy', incorrect_class)
np.save('/data0/jinhaibo/lixiaohao/ImageNet/FalsePos_label.npy', incorrect_label)