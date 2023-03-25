#%%
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from keras.models import load_model
model = load_model('/data0/jinhaibo/DGAN/train_model/lenet5.h5')
model.summary()
model.save_weights('/data0/jinhaibo/lixiaohao/adv_GTSRB/GTSRB_lenet5.h5')