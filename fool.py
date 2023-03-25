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
from keras.utils import to_categorical
tf.compat.v1.disable_eager_execution()
from keras.datasets import cifar10
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # （保证程序cuda序号与实际cuda序号对应）
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

#%%

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# x_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data.npy')
# x_train = x_train/255.0
# y_train = np.load('/data0/jinhaibo/DGAN/animals_10_datasets/vgg/train/img_data_label.npy')
# y_train = to_categorical(y_train, 10)
# x_class = x_train[np.argmax(y_train, axis=1)==9]
# y_class = y_train[np.argmax(y_train, axis=1)==9]
# x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
# y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
# y_train = to_categorical(y_train, 43)
# x_train = x_train /255.0
x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
y_train = to_categorical(y_train, 43)
x_train = x_train / 255.0
model = load_model('/data0/jinhaibo/DGAN/train_model/lenet5.h5')
# print(model.predict(x_train[0:1]))
# model.summary()

#%%
fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
attack = foolbox.attacks.PointwiseAttack(fmodel)


#%%
advs = []
label = []
for i in tqdm(range(10)):
    x = x_train[i]
    truth = np.argmax(y_train[i])
    print(x.shape)
    print(truth)
    x_adv = attack(x, truth)
    advs.append(x_adv)
    label.append(truth)


#%%
path = '/data0/jinhaibo/lixiaohao/adv_GTSRB/Lenet/PWA'
if not os.path.exists(path):
    os.makedirs(path)
np.save('/data0/jinhaibo/lixiaohao/adv_GTSRB/Lenet/PWA/adv_train.npy', advs)
np.save('/data0/jinhaibo/lixiaohao/adv_GTSRB/Lenet/PWA/adv_ori_label.npy', label)
advs = np.array(advs)
predictions = model.predict(advs)
accuracy = np.sum(np.argmax(predictions, axis=1) != label) / len(advs)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


#%%
# classifier = KerasClassifier(model=model, clip_values=(0, 1))
# attack = SquareAttack(estimator=classifier)
# x_train_adv = attack.generate(x_train, y_train)
# predictions = classifier.predict(x_train_adv)
# x_train_adv = x_train_adv[np.argmax(predictions, axis=1) != np.argmax(y_train, axis=1)]
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / 4300
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
# path = '/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/Square'
# if not os.path.exists(path):
#     os.makedirs(path)
# np.save('/data0/jinhaibo/lixiaohao/adv_GTSRB/ResNet20/Square/adv_train.npy', x_train_adv)