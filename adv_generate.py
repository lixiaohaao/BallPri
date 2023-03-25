import foolbox
import keras
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
# import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import load_model
import tensorflow as tf
from GTSRB import resnet_v1
tf.compat.v1.disable_eager_execution()
from keras.utils import to_categorical
from keras.datasets import cifar10
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

#%%

x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
y_train = to_categorical(y_train, 43)
x_train = x_train / 255.0
model = resnet_v1((48, 48, 3))
model.load_weights('/data0/jinhaibo/lixiaohao/adv_GTSRB/GTSRB_ResNet.h5')


#%%
fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
attack = foolbox.attacks.FGSM(fmodel)


#%%
advs = []
label = []
for i in tqdm(range(4300)):
    x = x_train[i]
    truth = np.argmax(y_train[i: i + 1], axis=1)
    x_adv = attack(x, truth)
    advs.append(x_adv)
    label.append(truth)

#%%
path = '/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/FGSM'
if not os.path.exists(path):
    os.makedirs(path)
np.save('/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/FGSM/adv.npy', advs)
np.save('/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/FGSM/adv_label.npy', label)
predictions = model.predict(advs)
accuracy = np.sum(np.argmax(predictions, axis=1) != label) / len(advs)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

