
# Convolutional neural networks for CIFAR100


```python
import matplotlib.pyplot as plt
%matplotlib notebook
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
```


```python
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model

import os
```

    Using TensorFlow backend.



```python
# Load the data: CIFAR100 with 20 class labels
(x_train_all, y_train_all), (x_test, y_test) = cifar100.load_data(label_mode='coarse')
num_classes = 20

# Convert class vectors to binary class matrices: we use the built-in Keras function for this
y_train_all = keras.utils.to_categorical(y_train_all, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

val_size = 6000
# make validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=val_size, 
                                                              stratify = y_train_all, random_state = 1)
# let's take a subset of the training data first, for playing around
x_train_small = x_train[:10000]
y_train_small = y_train[:10000]

x_train_all = x_train_all.astype('float32')
x_train = x_train.astype('float32')
x_train_small = x_train_small.astype('float32')
x_val = x_val.astype('float32')
x_test = x_test.astype('float32')
x_train_all /= 255.0
x_train /= 255.0
x_train_small /= 255.0
x_val /= 255.0
x_test /= 255

# Labels
labels = [
'aquatic mammals',
'fish',
'flowers',
'food containers',
'fruit and vegetables',
'household electrical devices',
'household furniture',
'insects',
'large carnivores',
'large man-made outdoor things',
'large natural outdoor scenes',
'large omnivores and herbivores',
'medium-sized mammals',
'non-insect invertebrates',
'people',
'reptiles',
'small mammals',
'trees',
'vehicles 1',
'vehicles 2'
]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_train_small.shape[0], 'small train samples')
print(x_val.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')
```

    x_train shape: (44000, 32, 32, 3)
    44000 train samples
    10000 small train samples
    6000 validation samples
    10000 test samples


## Model after Step3 - Stage 1


```python
test_model = Sequential()
#stack 1
test_model.add(Conv2D(96, (5, 5), padding='same',
                 input_shape=x_train.shape[1:]))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(96, (5, 5), padding='same'))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
#test_model.add(Dropout(0.1))

#stack2
test_model.add(Conv2D(128, (5, 5), padding='same'))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(128, (5, 5), padding='same'))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
#test_model.add(Dropout(0.2))

#stack3
test_model.add(Conv2D(256, (5, 5), padding='same'))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(256, (5, 5), padding='same'))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
#test_model.add(Dropout(0.2))

test_model.add(Flatten())
test_model.add(Dense(1024))
test_model.add(Activation('relu'))
#test_model.add(Dropout(0.5))
test_model.add(Dense(1024))
test_model.add(Activation('relu'))
#test_model.add(Dropout(0.5))

#output
test_model.add(Dense(num_classes))
test_model.add(Activation('softmax'))

test_model.summary()

epochs = 100
batch_size = 512

Adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
test_model.compile(loss='categorical_crossentropy',
              optimizer= Adam,
              metrics=['accuracy'])
filepath = 'step3_stage1.h5'

callbacks = [EarlyStopping(monitor='val_acc', patience=10),
             ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)]

print('Not using data augmentation!')
history = test_model.fit(x_train_small, y_train_small,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          shuffle=True, callbacks = callbacks)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_13 (Conv2D)           (None, 32, 32, 96)        7296      
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_18 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_14 (Conv2D)           (None, 32, 32, 96)        230496    
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_19 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 15, 15, 96)        0         
    _________________________________________________________________
    conv2d_15 (Conv2D)           (None, 15, 15, 128)       307328    
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_20 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 15, 15, 128)       409728    
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_21 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 7, 7, 256)         819456    
    _________________________________________________________________
    batch_normalization_17 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_22 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    conv2d_18 (Conv2D)           (None, 7, 7, 256)         1638656   
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_23 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 3, 3, 256)         0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 2304)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 1024)              2360320   
    _________________________________________________________________
    activation_24 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 1024)              1049600   
    _________________________________________________________________
    activation_25 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dense_9 (Dense)              (None, 20)                20500     
    _________________________________________________________________
    activation_26 (Activation)   (None, 20)                0         
    =================================================================
    Total params: 6,847,220
    Trainable params: 6,845,300
    Non-trainable params: 1,920
    _________________________________________________________________
    Not using data augmentation!
    Train on 10000 samples, validate on 6000 samples
    Epoch 1/100
    10000/10000 [==============================] - 6s 553us/step - loss: 2.7534 - acc: 0.1887 - val_loss: 2.4722 - val_acc: 0.2452
    
    Epoch 00001: val_acc improved from -inf to 0.24517, saving model to step3_stage1.h5
    Epoch 2/100
    10000/10000 [==============================] - 4s 356us/step - loss: 2.2076 - acc: 0.3276 - val_loss: 2.3009 - val_acc: 0.2930
    
    Epoch 00002: val_acc improved from 0.24517 to 0.29300, saving model to step3_stage1.h5
    Epoch 3/100
    10000/10000 [==============================] - 4s 361us/step - loss: 1.9061 - acc: 0.4255 - val_loss: 2.2185 - val_acc: 0.3170
    
    Epoch 00003: val_acc improved from 0.29300 to 0.31700, saving model to step3_stage1.h5
    Epoch 4/100
    10000/10000 [==============================] - 4s 361us/step - loss: 1.6360 - acc: 0.5033 - val_loss: 2.2007 - val_acc: 0.3243
    
    Epoch 00004: val_acc improved from 0.31700 to 0.32433, saving model to step3_stage1.h5
    Epoch 5/100
    10000/10000 [==============================] - 4s 362us/step - loss: 1.3428 - acc: 0.6118 - val_loss: 2.2605 - val_acc: 0.3182
    
    Epoch 00005: val_acc did not improve from 0.32433
    Epoch 6/100
    10000/10000 [==============================] - 4s 362us/step - loss: 1.0701 - acc: 0.6993 - val_loss: 2.0651 - val_acc: 0.3678
    
    Epoch 00006: val_acc improved from 0.32433 to 0.36783, saving model to step3_stage1.h5
    Epoch 7/100
    10000/10000 [==============================] - 4s 368us/step - loss: 0.7793 - acc: 0.8012 - val_loss: 2.2246 - val_acc: 0.3292
    
    Epoch 00007: val_acc did not improve from 0.36783
    Epoch 8/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.5516 - acc: 0.8825 - val_loss: 2.1668 - val_acc: 0.3592
    
    Epoch 00008: val_acc did not improve from 0.36783
    Epoch 9/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.3400 - acc: 0.9460 - val_loss: 2.3035 - val_acc: 0.3313
    
    Epoch 00009: val_acc did not improve from 0.36783
    Epoch 10/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.2043 - acc: 0.9774 - val_loss: 2.2570 - val_acc: 0.3573
    
    Epoch 00010: val_acc did not improve from 0.36783
    Epoch 11/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.1121 - acc: 0.9939 - val_loss: 2.3031 - val_acc: 0.3605
    
    Epoch 00011: val_acc did not improve from 0.36783
    Epoch 12/100
    10000/10000 [==============================] - 4s 365us/step - loss: 0.0548 - acc: 0.9996 - val_loss: 2.3053 - val_acc: 0.3773
    
    Epoch 00012: val_acc improved from 0.36783 to 0.37733, saving model to step3_stage1.h5
    Epoch 13/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.0289 - acc: 1.0000 - val_loss: 2.2364 - val_acc: 0.3968
    
    Epoch 00013: val_acc improved from 0.37733 to 0.39683, saving model to step3_stage1.h5
    Epoch 14/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0182 - acc: 1.0000 - val_loss: 2.2572 - val_acc: 0.4105
    
    Epoch 00014: val_acc improved from 0.39683 to 0.41050, saving model to step3_stage1.h5
    Epoch 15/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.0131 - acc: 1.0000 - val_loss: 2.2919 - val_acc: 0.4115
    
    Epoch 00015: val_acc improved from 0.41050 to 0.41150, saving model to step3_stage1.h5
    Epoch 16/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0106 - acc: 1.0000 - val_loss: 2.2903 - val_acc: 0.4210
    
    Epoch 00016: val_acc improved from 0.41150 to 0.42100, saving model to step3_stage1.h5
    Epoch 17/100
    10000/10000 [==============================] - 4s 361us/step - loss: 0.0090 - acc: 1.0000 - val_loss: 2.3130 - val_acc: 0.4212
    
    Epoch 00017: val_acc improved from 0.42100 to 0.42117, saving model to step3_stage1.h5
    Epoch 18/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.0076 - acc: 1.0000 - val_loss: 2.3204 - val_acc: 0.4255
    
    Epoch 00018: val_acc improved from 0.42117 to 0.42550, saving model to step3_stage1.h5
    Epoch 19/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.0066 - acc: 1.0000 - val_loss: 2.3668 - val_acc: 0.4263
    
    Epoch 00019: val_acc improved from 0.42550 to 0.42633, saving model to step3_stage1.h5
    Epoch 20/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.0057 - acc: 1.0000 - val_loss: 2.3798 - val_acc: 0.4257
    
    Epoch 00020: val_acc did not improve from 0.42633
    Epoch 21/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.0052 - acc: 1.0000 - val_loss: 2.4131 - val_acc: 0.4250
    
    Epoch 00021: val_acc did not improve from 0.42633
    Epoch 22/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0047 - acc: 1.0000 - val_loss: 2.4194 - val_acc: 0.4273
    
    Epoch 00022: val_acc improved from 0.42633 to 0.42733, saving model to step3_stage1.h5
    Epoch 23/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0042 - acc: 1.0000 - val_loss: 2.4474 - val_acc: 0.4265
    
    Epoch 00023: val_acc did not improve from 0.42733
    Epoch 24/100
    10000/10000 [==============================] - 4s 370us/step - loss: 0.0037 - acc: 1.0000 - val_loss: 2.4753 - val_acc: 0.4295
    
    Epoch 00024: val_acc improved from 0.42733 to 0.42950, saving model to step3_stage1.h5
    Epoch 25/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.0035 - acc: 1.0000 - val_loss: 2.5022 - val_acc: 0.4272
    
    Epoch 00025: val_acc did not improve from 0.42950
    Epoch 26/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0032 - acc: 1.0000 - val_loss: 2.5187 - val_acc: 0.4295
    
    Epoch 00026: val_acc improved from 0.42950 to 0.42950, saving model to step3_stage1.h5
    Epoch 27/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.0029 - acc: 1.0000 - val_loss: 2.5360 - val_acc: 0.4292
    
    Epoch 00027: val_acc did not improve from 0.42950
    Epoch 28/100
    10000/10000 [==============================] - 4s 362us/step - loss: 0.0026 - acc: 1.0000 - val_loss: 2.5526 - val_acc: 0.4285
    
    Epoch 00028: val_acc did not improve from 0.42950
    Epoch 29/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0024 - acc: 1.0000 - val_loss: 2.5713 - val_acc: 0.4295
    
    Epoch 00029: val_acc did not improve from 0.42950
    Epoch 30/100
    10000/10000 [==============================] - 4s 365us/step - loss: 0.0024 - acc: 1.0000 - val_loss: 2.5893 - val_acc: 0.4278
    
    Epoch 00030: val_acc did not improve from 0.42950
    Epoch 31/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0022 - acc: 1.0000 - val_loss: 2.6033 - val_acc: 0.4280
    
    Epoch 00031: val_acc did not improve from 0.42950
    Epoch 32/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0020 - acc: 1.0000 - val_loss: 2.6234 - val_acc: 0.4270
    
    Epoch 00032: val_acc did not improve from 0.42950
    Epoch 33/100
    10000/10000 [==============================] - 4s 364us/step - loss: 0.0018 - acc: 1.0000 - val_loss: 2.6345 - val_acc: 0.4293
    
    Epoch 00033: val_acc did not improve from 0.42950
    Epoch 34/100
    10000/10000 [==============================] - 4s 360us/step - loss: 0.0017 - acc: 1.0000 - val_loss: 2.6529 - val_acc: 0.4298
    
    Epoch 00034: val_acc improved from 0.42950 to 0.42983, saving model to step3_stage1.h5
    Epoch 35/100
    10000/10000 [==============================] - 4s 360us/step - loss: 0.0016 - acc: 1.0000 - val_loss: 2.6625 - val_acc: 0.4282
    
    Epoch 00035: val_acc did not improve from 0.42983
    Epoch 36/100
    10000/10000 [==============================] - 4s 360us/step - loss: 0.0014 - acc: 1.0000 - val_loss: 2.6798 - val_acc: 0.4283
    
    Epoch 00036: val_acc did not improve from 0.42983
    Epoch 37/100
    10000/10000 [==============================] - 4s 363us/step - loss: 0.0014 - acc: 1.0000 - val_loss: 2.6910 - val_acc: 0.4303
    
    Epoch 00037: val_acc improved from 0.42983 to 0.43033, saving model to step3_stage1.h5
    Epoch 38/100
    10000/10000 [==============================] - 4s 361us/step - loss: 0.0013 - acc: 1.0000 - val_loss: 2.7047 - val_acc: 0.4295
    
    Epoch 00038: val_acc did not improve from 0.43033
    Epoch 39/100
    10000/10000 [==============================] - 4s 361us/step - loss: 0.0012 - acc: 1.0000 - val_loss: 2.7217 - val_acc: 0.4302
    
    Epoch 00039: val_acc did not improve from 0.43033
    Epoch 40/100
    10000/10000 [==============================] - 4s 360us/step - loss: 0.0011 - acc: 1.0000 - val_loss: 2.7328 - val_acc: 0.4285
    
    Epoch 00040: val_acc did not improve from 0.43033
    Epoch 41/100
    10000/10000 [==============================] - 4s 369us/step - loss: 0.0011 - acc: 1.0000 - val_loss: 2.7398 - val_acc: 0.4293
    
    Epoch 00041: val_acc did not improve from 0.43033
    Epoch 42/100
    10000/10000 [==============================] - 4s 361us/step - loss: 0.0011 - acc: 1.0000 - val_loss: 2.7532 - val_acc: 0.4287
    
    Epoch 00042: val_acc did not improve from 0.43033
    Epoch 43/100
    10000/10000 [==============================] - 4s 361us/step - loss: 9.6953e-04 - acc: 1.0000 - val_loss: 2.7639 - val_acc: 0.4290
    
    Epoch 00043: val_acc did not improve from 0.43033
    Epoch 44/100
    10000/10000 [==============================] - 4s 365us/step - loss: 8.7422e-04 - acc: 1.0000 - val_loss: 2.7785 - val_acc: 0.4298
    
    Epoch 00044: val_acc did not improve from 0.43033
    Epoch 45/100
    10000/10000 [==============================] - 4s 365us/step - loss: 8.5376e-04 - acc: 1.0000 - val_loss: 2.7833 - val_acc: 0.4277
    
    Epoch 00045: val_acc did not improve from 0.43033
    Epoch 46/100
    10000/10000 [==============================] - 4s 361us/step - loss: 8.1486e-04 - acc: 1.0000 - val_loss: 2.7994 - val_acc: 0.4275
    
    Epoch 00046: val_acc did not improve from 0.43033
    Epoch 47/100
    10000/10000 [==============================] - 4s 363us/step - loss: 7.6257e-04 - acc: 1.0000 - val_loss: 2.8079 - val_acc: 0.4313
    
    Epoch 00047: val_acc improved from 0.43033 to 0.43133, saving model to step3_stage1.h5
    Epoch 48/100
    10000/10000 [==============================] - 4s 361us/step - loss: 7.5284e-04 - acc: 1.0000 - val_loss: 2.8176 - val_acc: 0.4305
    
    Epoch 00048: val_acc did not improve from 0.43133
    Epoch 49/100
    10000/10000 [==============================] - 4s 361us/step - loss: 7.0300e-04 - acc: 1.0000 - val_loss: 2.8289 - val_acc: 0.4300
    
    Epoch 00049: val_acc did not improve from 0.43133
    Epoch 50/100
    10000/10000 [==============================] - 4s 361us/step - loss: 6.9584e-04 - acc: 1.0000 - val_loss: 2.8378 - val_acc: 0.4293
    
    Epoch 00050: val_acc did not improve from 0.43133
    Epoch 51/100
    10000/10000 [==============================] - 4s 360us/step - loss: 6.3593e-04 - acc: 1.0000 - val_loss: 2.8465 - val_acc: 0.4288
    
    Epoch 00051: val_acc did not improve from 0.43133
    Epoch 52/100
    10000/10000 [==============================] - 4s 360us/step - loss: 6.0578e-04 - acc: 1.0000 - val_loss: 2.8578 - val_acc: 0.4292
    
    Epoch 00052: val_acc did not improve from 0.43133
    Epoch 53/100
    10000/10000 [==============================] - 4s 360us/step - loss: 6.2471e-04 - acc: 1.0000 - val_loss: 2.8685 - val_acc: 0.4305
    
    Epoch 00053: val_acc did not improve from 0.43133
    Epoch 54/100
    10000/10000 [==============================] - 4s 361us/step - loss: 5.6745e-04 - acc: 1.0000 - val_loss: 2.8769 - val_acc: 0.4282
    
    Epoch 00054: val_acc did not improve from 0.43133
    Epoch 55/100
    10000/10000 [==============================] - 4s 361us/step - loss: 5.3916e-04 - acc: 1.0000 - val_loss: 2.8885 - val_acc: 0.4288
    
    Epoch 00055: val_acc did not improve from 0.43133
    Epoch 56/100
    10000/10000 [==============================] - 4s 361us/step - loss: 5.2491e-04 - acc: 1.0000 - val_loss: 2.8963 - val_acc: 0.4293
    
    Epoch 00056: val_acc did not improve from 0.43133
    Epoch 57/100
    10000/10000 [==============================] - 4s 361us/step - loss: 4.8929e-04 - acc: 1.0000 - val_loss: 2.9037 - val_acc: 0.4300
    
    Epoch 00057: val_acc did not improve from 0.43133



```python
# Score trained model.
test_model.load_weights('step3_stage1.h5')
train_scores = test_model.evaluate(x_train_small, y_train_small, verbose=1)
val_scores = test_model.evaluate(x_val, y_val, verbose=1)

print('Training loss:', train_scores[0],', training accuracy: ',train_scores[1])
print('Validation loss:', val_scores[0],', validation accuracy: ',val_scores[1])

real_epochs = len(history.history['acc'])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.arange(1,real_epochs+1,1),history.history['acc'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_acc'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend()

# Note: loss is the loss function that is optimized for multi-class classification
# i.e. the multi-class version of cross-entropy error
plt.subplot(1,2,2)
plt.plot(np.arange(1,real_epochs+1,1),history.history['loss'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_loss'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

    10000/10000 [==============================] - 2s 232us/step
    6000/6000 [==============================] - 1s 224us/step
    Training loss: 0.000668073431682 , training accuracy:  1.0
    Validation loss: 2.80791947269 , validation accuracy:  0.431333333333



![png](output_6_1.png)


Model is powerfull enough.

Learning convergence.

But heavily overfitting.

## Model Step1 - Stage 2 
*before using dynamic learning rate*


```python
test_model = Sequential()
#stack 1
test_model.add(Conv2D(96, (5, 5), padding='same',
                 input_shape=x_train.shape[1:], kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(96, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.1))

#stack2
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

#stack3
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

test_model.add(Flatten())
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))

#output
test_model.add(Dense(num_classes))
test_model.add(Activation('softmax'))

test_model.summary()

epochs = 200
batch_size = 512

Adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
test_model.compile(loss='categorical_crossentropy',
              optimizer= Adam,
              metrics=['accuracy'])
filepath = 'step1_stage2_beforedylr.h5'

callbacks = [EarlyStopping(monitor='val_acc', patience=20),
             ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)]

print('Not using data augmentation!')
history = test_model.fit(x_train_small, y_train_small,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          shuffle=True, callbacks = callbacks)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_43 (Conv2D)           (None, 32, 32, 96)        7296      
    _________________________________________________________________
    batch_normalization_43 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_63 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_44 (Conv2D)           (None, 32, 32, 96)        230496    
    _________________________________________________________________
    batch_normalization_44 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_64 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    max_pooling2d_22 (MaxPooling (None, 15, 15, 96)        0         
    _________________________________________________________________
    dropout_22 (Dropout)         (None, 15, 15, 96)        0         
    _________________________________________________________________
    conv2d_45 (Conv2D)           (None, 15, 15, 128)       307328    
    _________________________________________________________________
    batch_normalization_45 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_65 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    conv2d_46 (Conv2D)           (None, 15, 15, 128)       409728    
    _________________________________________________________________
    batch_normalization_46 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_66 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    max_pooling2d_23 (MaxPooling (None, 7, 7, 128)         0         
    _________________________________________________________________
    dropout_23 (Dropout)         (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_47 (Conv2D)           (None, 7, 7, 256)         819456    
    _________________________________________________________________
    batch_normalization_47 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_67 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    conv2d_48 (Conv2D)           (None, 7, 7, 256)         1638656   
    _________________________________________________________________
    batch_normalization_48 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_68 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    max_pooling2d_24 (MaxPooling (None, 3, 3, 256)         0         
    _________________________________________________________________
    dropout_24 (Dropout)         (None, 3, 3, 256)         0         
    _________________________________________________________________
    flatten_8 (Flatten)          (None, 2304)              0         
    _________________________________________________________________
    dense_22 (Dense)             (None, 1024)              2360320   
    _________________________________________________________________
    activation_69 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dropout_25 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 1024)              1049600   
    _________________________________________________________________
    activation_70 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dropout_26 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_24 (Dense)             (None, 20)                20500     
    _________________________________________________________________
    activation_71 (Activation)   (None, 20)                0         
    =================================================================
    Total params: 6,847,220
    Trainable params: 6,845,300
    Non-trainable params: 1,920
    _________________________________________________________________
    Not using data augmentation!
    Train on 10000 samples, validate on 6000 samples
    Epoch 1/200
    10000/10000 [==============================] - 7s 699us/step - loss: 4.0113 - acc: 0.0599 - val_loss: 2.8365 - val_acc: 0.1370
    
    Epoch 00001: val_acc improved from -inf to 0.13700, saving model to step1_stage2_beforedylr.h5
    Epoch 2/200
    10000/10000 [==============================] - 4s 369us/step - loss: 3.0345 - acc: 0.0893 - val_loss: 2.7733 - val_acc: 0.1745
    
    Epoch 00002: val_acc improved from 0.13700 to 0.17450, saving model to step1_stage2_beforedylr.h5
    Epoch 3/200
    10000/10000 [==============================] - 4s 373us/step - loss: 2.9061 - acc: 0.1068 - val_loss: 2.7180 - val_acc: 0.1855
    
    Epoch 00003: val_acc improved from 0.17450 to 0.18550, saving model to step1_stage2_beforedylr.h5
    Epoch 4/200
    10000/10000 [==============================] - 4s 373us/step - loss: 2.8454 - acc: 0.1227 - val_loss: 2.6702 - val_acc: 0.1932
    
    Epoch 00004: val_acc improved from 0.18550 to 0.19317, saving model to step1_stage2_beforedylr.h5
    Epoch 5/200
    10000/10000 [==============================] - 4s 380us/step - loss: 2.7950 - acc: 0.1373 - val_loss: 2.6288 - val_acc: 0.1978
    
    Epoch 00005: val_acc improved from 0.19317 to 0.19783, saving model to step1_stage2_beforedylr.h5
    Epoch 6/200
    10000/10000 [==============================] - 4s 377us/step - loss: 2.7422 - acc: 0.1520 - val_loss: 2.6008 - val_acc: 0.2122
    
    Epoch 00006: val_acc improved from 0.19783 to 0.21217, saving model to step1_stage2_beforedylr.h5
    Epoch 7/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.7215 - acc: 0.1564 - val_loss: 2.5855 - val_acc: 0.2110
    
    Epoch 00007: val_acc did not improve from 0.21217
    Epoch 8/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.6706 - acc: 0.1686 - val_loss: 2.5662 - val_acc: 0.2188
    
    Epoch 00008: val_acc improved from 0.21217 to 0.21883, saving model to step1_stage2_beforedylr.h5
    Epoch 9/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.6451 - acc: 0.1762 - val_loss: 2.5489 - val_acc: 0.2180
    
    Epoch 00009: val_acc did not improve from 0.21883
    Epoch 10/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.6113 - acc: 0.1931 - val_loss: 2.5143 - val_acc: 0.2270
    
    Epoch 00010: val_acc improved from 0.21883 to 0.22700, saving model to step1_stage2_beforedylr.h5
    Epoch 11/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.5850 - acc: 0.1977 - val_loss: 2.4972 - val_acc: 0.2388
    
    Epoch 00011: val_acc improved from 0.22700 to 0.23883, saving model to step1_stage2_beforedylr.h5
    Epoch 12/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.5436 - acc: 0.2148 - val_loss: 2.5153 - val_acc: 0.2182
    
    Epoch 00012: val_acc did not improve from 0.23883
    Epoch 13/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.5299 - acc: 0.2125 - val_loss: 2.4602 - val_acc: 0.2380
    
    Epoch 00013: val_acc did not improve from 0.23883
    Epoch 14/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.5012 - acc: 0.2201 - val_loss: 2.4342 - val_acc: 0.2432
    
    Epoch 00014: val_acc improved from 0.23883 to 0.24317, saving model to step1_stage2_beforedylr.h5
    Epoch 15/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.4759 - acc: 0.2310 - val_loss: 2.4097 - val_acc: 0.2592
    
    Epoch 00015: val_acc improved from 0.24317 to 0.25917, saving model to step1_stage2_beforedylr.h5
    Epoch 16/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.4516 - acc: 0.2411 - val_loss: 2.3735 - val_acc: 0.2750
    
    Epoch 00016: val_acc improved from 0.25917 to 0.27500, saving model to step1_stage2_beforedylr.h5
    Epoch 17/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.4222 - acc: 0.2413 - val_loss: 2.3480 - val_acc: 0.2780
    
    Epoch 00017: val_acc improved from 0.27500 to 0.27800, saving model to step1_stage2_beforedylr.h5
    Epoch 18/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.4003 - acc: 0.2558 - val_loss: 2.3558 - val_acc: 0.2767
    
    Epoch 00018: val_acc did not improve from 0.27800
    Epoch 19/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.3732 - acc: 0.2553 - val_loss: 2.3758 - val_acc: 0.2715
    
    Epoch 00019: val_acc did not improve from 0.27800
    Epoch 20/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.3564 - acc: 0.2655 - val_loss: 2.3664 - val_acc: 0.2728
    
    Epoch 00020: val_acc did not improve from 0.27800
    Epoch 21/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.3117 - acc: 0.2742 - val_loss: 2.3041 - val_acc: 0.2878
    
    Epoch 00021: val_acc improved from 0.27800 to 0.28783, saving model to step1_stage2_beforedylr.h5
    Epoch 22/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.3083 - acc: 0.2794 - val_loss: 2.2704 - val_acc: 0.3062
    
    Epoch 00022: val_acc improved from 0.28783 to 0.30617, saving model to step1_stage2_beforedylr.h5
    Epoch 23/200
    10000/10000 [==============================] - 4s 376us/step - loss: 2.2773 - acc: 0.2936 - val_loss: 2.3663 - val_acc: 0.2745
    
    Epoch 00023: val_acc did not improve from 0.30617
    Epoch 24/200
    10000/10000 [==============================] - 4s 377us/step - loss: 2.2515 - acc: 0.2977 - val_loss: 2.2589 - val_acc: 0.3043
    
    Epoch 00024: val_acc did not improve from 0.30617
    Epoch 25/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.2159 - acc: 0.3024 - val_loss: 2.2703 - val_acc: 0.2922
    
    Epoch 00025: val_acc did not improve from 0.30617
    Epoch 26/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.1902 - acc: 0.3160 - val_loss: 2.2107 - val_acc: 0.3182
    
    Epoch 00026: val_acc improved from 0.30617 to 0.31817, saving model to step1_stage2_beforedylr.h5
    Epoch 27/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.1758 - acc: 0.3187 - val_loss: 2.2066 - val_acc: 0.3185
    
    Epoch 00027: val_acc improved from 0.31817 to 0.31850, saving model to step1_stage2_beforedylr.h5
    Epoch 28/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.1464 - acc: 0.3281 - val_loss: 2.1518 - val_acc: 0.3282
    
    Epoch 00028: val_acc improved from 0.31850 to 0.32817, saving model to step1_stage2_beforedylr.h5
    Epoch 29/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.1276 - acc: 0.3359 - val_loss: 2.1684 - val_acc: 0.3293
    
    Epoch 00029: val_acc improved from 0.32817 to 0.32933, saving model to step1_stage2_beforedylr.h5
    Epoch 30/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.1099 - acc: 0.3384 - val_loss: 2.1284 - val_acc: 0.3372
    
    Epoch 00030: val_acc improved from 0.32933 to 0.33717, saving model to step1_stage2_beforedylr.h5
    Epoch 31/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.0688 - acc: 0.3452 - val_loss: 2.1170 - val_acc: 0.3450
    
    Epoch 00031: val_acc improved from 0.33717 to 0.34500, saving model to step1_stage2_beforedylr.h5
    Epoch 32/200
    10000/10000 [==============================] - 4s 375us/step - loss: 2.0398 - acc: 0.3596 - val_loss: 2.1281 - val_acc: 0.3415
    
    Epoch 00032: val_acc did not improve from 0.34500
    Epoch 33/200
    10000/10000 [==============================] - 4s 374us/step - loss: 2.0271 - acc: 0.3626 - val_loss: 2.1103 - val_acc: 0.3455
    
    Epoch 00033: val_acc improved from 0.34500 to 0.34550, saving model to step1_stage2_beforedylr.h5
    Epoch 34/200
    10000/10000 [==============================] - 4s 374us/step - loss: 1.9919 - acc: 0.3798 - val_loss: 2.2414 - val_acc: 0.3203
    
    Epoch 00034: val_acc did not improve from 0.34550
    Epoch 35/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.9866 - acc: 0.3781 - val_loss: 2.2144 - val_acc: 0.3293
    
    Epoch 00035: val_acc did not improve from 0.34550
    Epoch 36/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.9448 - acc: 0.3902 - val_loss: 2.1231 - val_acc: 0.3468
    
    Epoch 00036: val_acc improved from 0.34550 to 0.34683, saving model to step1_stage2_beforedylr.h5
    Epoch 37/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.9370 - acc: 0.3861 - val_loss: 2.1153 - val_acc: 0.3593
    
    Epoch 00037: val_acc improved from 0.34683 to 0.35933, saving model to step1_stage2_beforedylr.h5
    Epoch 38/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.9190 - acc: 0.3922 - val_loss: 2.2786 - val_acc: 0.3260
    
    Epoch 00038: val_acc did not improve from 0.35933
    Epoch 39/200
    10000/10000 [==============================] - 4s 380us/step - loss: 1.8916 - acc: 0.4068 - val_loss: 2.0086 - val_acc: 0.3828
    
    Epoch 00039: val_acc improved from 0.35933 to 0.38283, saving model to step1_stage2_beforedylr.h5
    Epoch 40/200
    10000/10000 [==============================] - 4s 379us/step - loss: 1.8708 - acc: 0.4094 - val_loss: 2.1029 - val_acc: 0.3583
    
    Epoch 00040: val_acc did not improve from 0.38283
    Epoch 41/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.8574 - acc: 0.4179 - val_loss: 2.0068 - val_acc: 0.3857
    
    Epoch 00041: val_acc improved from 0.38283 to 0.38567, saving model to step1_stage2_beforedylr.h5
    Epoch 42/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.8157 - acc: 0.4247 - val_loss: 2.1109 - val_acc: 0.3645
    
    Epoch 00042: val_acc did not improve from 0.38567
    Epoch 43/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.8227 - acc: 0.4232 - val_loss: 2.2490 - val_acc: 0.3480
    
    Epoch 00043: val_acc did not improve from 0.38567
    Epoch 44/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.7891 - acc: 0.4331 - val_loss: 2.2157 - val_acc: 0.3488
    
    Epoch 00044: val_acc did not improve from 0.38567
    Epoch 45/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.7671 - acc: 0.4461 - val_loss: 2.0384 - val_acc: 0.3795
    
    Epoch 00045: val_acc did not improve from 0.38567
    Epoch 46/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.7498 - acc: 0.4446 - val_loss: 2.1732 - val_acc: 0.3593
    
    Epoch 00046: val_acc did not improve from 0.38567
    Epoch 47/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.7282 - acc: 0.4507 - val_loss: 2.1541 - val_acc: 0.3632
    
    Epoch 00047: val_acc did not improve from 0.38567
    Epoch 48/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.7060 - acc: 0.4549 - val_loss: 1.9482 - val_acc: 0.4095
    
    Epoch 00048: val_acc improved from 0.38567 to 0.40950, saving model to step1_stage2_beforedylr.h5
    Epoch 49/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.6887 - acc: 0.4620 - val_loss: 2.1858 - val_acc: 0.3670
    
    Epoch 00049: val_acc did not improve from 0.40950
    Epoch 50/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.6685 - acc: 0.4696 - val_loss: 2.0719 - val_acc: 0.3892
    
    Epoch 00050: val_acc did not improve from 0.40950
    Epoch 51/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.6360 - acc: 0.4784 - val_loss: 1.9823 - val_acc: 0.3993
    
    Epoch 00051: val_acc did not improve from 0.40950
    Epoch 52/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.6278 - acc: 0.4798 - val_loss: 2.0998 - val_acc: 0.3822
    
    Epoch 00052: val_acc did not improve from 0.40950
    Epoch 53/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.5966 - acc: 0.4921 - val_loss: 2.0248 - val_acc: 0.3938
    
    Epoch 00053: val_acc did not improve from 0.40950
    Epoch 54/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.6027 - acc: 0.4893 - val_loss: 1.9006 - val_acc: 0.4172
    
    Epoch 00054: val_acc improved from 0.40950 to 0.41717, saving model to step1_stage2_beforedylr.h5
    Epoch 55/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.5693 - acc: 0.4940 - val_loss: 2.0370 - val_acc: 0.3900
    
    Epoch 00055: val_acc did not improve from 0.41717
    Epoch 56/200
    10000/10000 [==============================] - 4s 378us/step - loss: 1.5491 - acc: 0.5066 - val_loss: 1.9176 - val_acc: 0.4180
    
    Epoch 00056: val_acc improved from 0.41717 to 0.41800, saving model to step1_stage2_beforedylr.h5
    Epoch 57/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.5495 - acc: 0.5022 - val_loss: 2.0729 - val_acc: 0.3972
    
    Epoch 00057: val_acc did not improve from 0.41800
    Epoch 58/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.5107 - acc: 0.5116 - val_loss: 2.0266 - val_acc: 0.4108
    
    Epoch 00058: val_acc did not improve from 0.41800
    Epoch 59/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.4942 - acc: 0.5242 - val_loss: 2.1299 - val_acc: 0.3885
    
    Epoch 00059: val_acc did not improve from 0.41800
    Epoch 60/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.4718 - acc: 0.5262 - val_loss: 2.3353 - val_acc: 0.3683
    
    Epoch 00060: val_acc did not improve from 0.41800
    Epoch 61/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.4467 - acc: 0.5333 - val_loss: 2.5409 - val_acc: 0.3508
    
    Epoch 00061: val_acc did not improve from 0.41800
    Epoch 62/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.4322 - acc: 0.5437 - val_loss: 2.1457 - val_acc: 0.3993
    
    Epoch 00062: val_acc did not improve from 0.41800
    Epoch 63/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.4241 - acc: 0.5411 - val_loss: 2.4886 - val_acc: 0.3473
    
    Epoch 00063: val_acc did not improve from 0.41800
    Epoch 64/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.3969 - acc: 0.5533 - val_loss: 2.0148 - val_acc: 0.4245
    
    Epoch 00064: val_acc improved from 0.41800 to 0.42450, saving model to step1_stage2_beforedylr.h5
    Epoch 65/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.3647 - acc: 0.5547 - val_loss: 2.5523 - val_acc: 0.3703
    
    Epoch 00065: val_acc did not improve from 0.42450
    Epoch 66/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.3666 - acc: 0.5613 - val_loss: 2.1788 - val_acc: 0.4000
    
    Epoch 00066: val_acc did not improve from 0.42450
    Epoch 67/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.3599 - acc: 0.5560 - val_loss: 2.4460 - val_acc: 0.3663
    
    Epoch 00067: val_acc did not improve from 0.42450
    Epoch 68/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.3295 - acc: 0.5714 - val_loss: 2.2458 - val_acc: 0.4115
    
    Epoch 00068: val_acc did not improve from 0.42450
    Epoch 69/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.3293 - acc: 0.5701 - val_loss: 2.3945 - val_acc: 0.3832
    
    Epoch 00069: val_acc did not improve from 0.42450
    Epoch 70/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.3125 - acc: 0.5714 - val_loss: 1.9632 - val_acc: 0.4490
    
    Epoch 00070: val_acc improved from 0.42450 to 0.44900, saving model to step1_stage2_beforedylr.h5
    Epoch 71/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.2782 - acc: 0.5860 - val_loss: 2.1505 - val_acc: 0.4125
    
    Epoch 00071: val_acc did not improve from 0.44900
    Epoch 72/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.2511 - acc: 0.5945 - val_loss: 1.9175 - val_acc: 0.4437
    
    Epoch 00072: val_acc did not improve from 0.44900
    Epoch 73/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.2452 - acc: 0.5931 - val_loss: 2.3215 - val_acc: 0.3975
    
    Epoch 00073: val_acc did not improve from 0.44900
    Epoch 74/200
    10000/10000 [==============================] - 4s 378us/step - loss: 1.2560 - acc: 0.5924 - val_loss: 2.0310 - val_acc: 0.4383
    
    Epoch 00074: val_acc did not improve from 0.44900
    Epoch 75/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.2094 - acc: 0.6132 - val_loss: 1.8365 - val_acc: 0.4763
    
    Epoch 00075: val_acc improved from 0.44900 to 0.47633, saving model to step1_stage2_beforedylr.h5
    Epoch 76/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.1940 - acc: 0.6142 - val_loss: 1.9044 - val_acc: 0.4608
    
    Epoch 00076: val_acc did not improve from 0.47633
    Epoch 77/200
    10000/10000 [==============================] - 4s 375us/step - loss: 1.1635 - acc: 0.6208 - val_loss: 2.1580 - val_acc: 0.4333
    
    Epoch 00077: val_acc did not improve from 0.47633
    Epoch 78/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.1460 - acc: 0.6279 - val_loss: 2.3198 - val_acc: 0.4085
    
    Epoch 00078: val_acc did not improve from 0.47633
    Epoch 79/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.1256 - acc: 0.6343 - val_loss: 2.1927 - val_acc: 0.4278
    
    Epoch 00079: val_acc did not improve from 0.47633
    Epoch 80/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.1191 - acc: 0.6318 - val_loss: 1.9160 - val_acc: 0.4678
    
    Epoch 00080: val_acc did not improve from 0.47633
    Epoch 81/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.1067 - acc: 0.6370 - val_loss: 1.9735 - val_acc: 0.4668
    
    Epoch 00081: val_acc did not improve from 0.47633
    Epoch 82/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.1016 - acc: 0.6374 - val_loss: 1.9465 - val_acc: 0.4617
    
    Epoch 00082: val_acc did not improve from 0.47633
    Epoch 83/200
    10000/10000 [==============================] - 4s 378us/step - loss: 1.0952 - acc: 0.6399 - val_loss: 2.0889 - val_acc: 0.4428
    
    Epoch 00083: val_acc did not improve from 0.47633
    Epoch 84/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.0587 - acc: 0.6477 - val_loss: 2.4904 - val_acc: 0.4023
    
    Epoch 00084: val_acc did not improve from 0.47633
    Epoch 85/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.0557 - acc: 0.6567 - val_loss: 2.2439 - val_acc: 0.4338
    
    Epoch 00085: val_acc did not improve from 0.47633
    Epoch 86/200
    10000/10000 [==============================] - 4s 376us/step - loss: 1.0270 - acc: 0.6624 - val_loss: 2.1245 - val_acc: 0.4470
    
    Epoch 00086: val_acc did not improve from 0.47633
    Epoch 87/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.0031 - acc: 0.6641 - val_loss: 2.1294 - val_acc: 0.4485
    
    Epoch 00087: val_acc did not improve from 0.47633
    Epoch 88/200
    10000/10000 [==============================] - 4s 377us/step - loss: 1.0004 - acc: 0.6632 - val_loss: 2.0250 - val_acc: 0.4713
    
    Epoch 00088: val_acc did not improve from 0.47633
    Epoch 89/200
    10000/10000 [==============================] - 4s 376us/step - loss: 0.9741 - acc: 0.6730 - val_loss: 2.2952 - val_acc: 0.4300
    
    Epoch 00089: val_acc did not improve from 0.47633
    Epoch 90/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.9667 - acc: 0.6775 - val_loss: 2.2488 - val_acc: 0.4392
    
    Epoch 00090: val_acc did not improve from 0.47633
    Epoch 91/200
    10000/10000 [==============================] - 4s 377us/step - loss: 0.9464 - acc: 0.6857 - val_loss: 2.3372 - val_acc: 0.4232
    
    Epoch 00091: val_acc did not improve from 0.47633
    Epoch 92/200
    10000/10000 [==============================] - 4s 378us/step - loss: 0.9472 - acc: 0.6853 - val_loss: 2.0463 - val_acc: 0.4597
    
    Epoch 00092: val_acc did not improve from 0.47633
    Epoch 93/200
    10000/10000 [==============================] - 4s 377us/step - loss: 0.9105 - acc: 0.6952 - val_loss: 2.2626 - val_acc: 0.4313
    
    Epoch 00093: val_acc did not improve from 0.47633
    Epoch 94/200
    10000/10000 [==============================] - 4s 377us/step - loss: 0.9055 - acc: 0.7032 - val_loss: 2.2188 - val_acc: 0.4507
    
    Epoch 00094: val_acc did not improve from 0.47633
    Epoch 95/200
    10000/10000 [==============================] - 4s 377us/step - loss: 0.8962 - acc: 0.7035 - val_loss: 2.5726 - val_acc: 0.4112
    
    Epoch 00095: val_acc did not improve from 0.47633



```python
# Score trained model.
test_model.load_weights('step1_stage2_beforedylr.h5')
train_scores = test_model.evaluate(x_train_small, y_train_small, verbose=1)
val_scores = test_model.evaluate(x_val, y_val, verbose=1)

print('Training loss:', train_scores[0],', training accuracy: ',train_scores[1])
print('Validation loss:', val_scores[0],', validation accuracy: ',val_scores[1])

real_epochs = len(history.history['acc'])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.arange(1,real_epochs+1,1),history.history['acc'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_acc'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend()

# Note: loss is the loss function that is optimized for multi-class classification
# i.e. the multi-class version of cross-entropy error
plt.subplot(1,2,2)
plt.plot(np.arange(1,real_epochs+1,1),history.history['loss'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_loss'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

    10000/10000 [==============================] - 2s 247us/step
    6000/6000 [==============================] - 1s 245us/step
    Training loss: 0.936414838028 , training accuracy:  0.6977
    Validation loss: 1.83650236416 , validation accuracy:  0.476333333333



![png](output_10_1.png)


The curve is bumpy!!!!
Not enought powerful. 
Overfitting is reduced.

## Model with Dynamic Learning Rate (Step1 - Stage2)


```python
def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 40:
        lrate = 0.00005
    if epoch > 70:
        lrate = 0.000005
    return lrate

test_model = Sequential()
#stack 1
test_model.add(Conv2D(96, (5, 5), padding='same',
                 input_shape=x_train.shape[1:], kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(96, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.1))

#stack2
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

#stack3
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

test_model.add(Flatten())
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))

#output
test_model.add(Dense(num_classes))
test_model.add(Activation('softmax'))

test_model.summary()

epochs = 200
batch_size = 512

Adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
test_model.compile(loss='categorical_crossentropy',
              optimizer= Adam,
              metrics=['accuracy'])
filepath = 'step1_stage2_afterdylr.h5'

callbacks = [EarlyStopping(monitor='val_acc', patience=20),
             ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True), 
             LearningRateScheduler(lr_schedule)]

print('Not using data augmentation!')
history = test_model.fit(x_train_small, y_train_small,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          shuffle=True, callbacks = callbacks)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_61 (Conv2D)           (None, 32, 32, 96)        7296      
    _________________________________________________________________
    batch_normalization_61 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_90 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_62 (Conv2D)           (None, 32, 32, 96)        230496    
    _________________________________________________________________
    batch_normalization_62 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_91 (Activation)   (None, 32, 32, 96)        0         
    _________________________________________________________________
    max_pooling2d_31 (MaxPooling (None, 15, 15, 96)        0         
    _________________________________________________________________
    dropout_37 (Dropout)         (None, 15, 15, 96)        0         
    _________________________________________________________________
    conv2d_63 (Conv2D)           (None, 15, 15, 128)       307328    
    _________________________________________________________________
    batch_normalization_63 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_92 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    conv2d_64 (Conv2D)           (None, 15, 15, 128)       409728    
    _________________________________________________________________
    batch_normalization_64 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_93 (Activation)   (None, 15, 15, 128)       0         
    _________________________________________________________________
    max_pooling2d_32 (MaxPooling (None, 7, 7, 128)         0         
    _________________________________________________________________
    dropout_38 (Dropout)         (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_65 (Conv2D)           (None, 7, 7, 256)         819456    
    _________________________________________________________________
    batch_normalization_65 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_94 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    conv2d_66 (Conv2D)           (None, 7, 7, 256)         1638656   
    _________________________________________________________________
    batch_normalization_66 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_95 (Activation)   (None, 7, 7, 256)         0         
    _________________________________________________________________
    max_pooling2d_33 (MaxPooling (None, 3, 3, 256)         0         
    _________________________________________________________________
    dropout_39 (Dropout)         (None, 3, 3, 256)         0         
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 2304)              0         
    _________________________________________________________________
    dense_31 (Dense)             (None, 1024)              2360320   
    _________________________________________________________________
    activation_96 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dropout_40 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_32 (Dense)             (None, 1024)              1049600   
    _________________________________________________________________
    activation_97 (Activation)   (None, 1024)              0         
    _________________________________________________________________
    dropout_41 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_33 (Dense)             (None, 20)                20500     
    _________________________________________________________________
    activation_98 (Activation)   (None, 20)                0         
    =================================================================
    Total params: 6,847,220
    Trainable params: 6,845,300
    Non-trainable params: 1,920
    _________________________________________________________________
    Not using data augmentation!
    Train on 10000 samples, validate on 6000 samples
    Epoch 1/200
    10000/10000 [==============================] - 11s 1ms/step - loss: 3.2841 - acc: 0.0783 - val_loss: 3.1345 - val_acc: 0.1150
    
    Epoch 00001: val_acc improved from -inf to 0.11500, saving model to step1_stage2_afterdylr.h5
    Epoch 2/200
    10000/10000 [==============================] - 4s 391us/step - loss: 2.8263 - acc: 0.1256 - val_loss: 2.7804 - val_acc: 0.1340
    
    Epoch 00002: val_acc improved from 0.11500 to 0.13400, saving model to step1_stage2_afterdylr.h5
    Epoch 3/200
    10000/10000 [==============================] - 4s 382us/step - loss: 2.7257 - acc: 0.1446 - val_loss: 2.7019 - val_acc: 0.1328
    
    Epoch 00003: val_acc did not improve from 0.13400
    Epoch 4/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.6275 - acc: 0.1742 - val_loss: 2.7038 - val_acc: 0.1518
    
    Epoch 00004: val_acc improved from 0.13400 to 0.15183, saving model to step1_stage2_afterdylr.h5
    Epoch 5/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.5499 - acc: 0.1989 - val_loss: 2.6790 - val_acc: 0.1578
    
    Epoch 00005: val_acc improved from 0.15183 to 0.15783, saving model to step1_stage2_afterdylr.h5
    Epoch 6/200
    10000/10000 [==============================] - 4s 382us/step - loss: 2.4841 - acc: 0.2254 - val_loss: 2.5706 - val_acc: 0.2070
    
    Epoch 00006: val_acc improved from 0.15783 to 0.20700, saving model to step1_stage2_afterdylr.h5
    Epoch 7/200
    10000/10000 [==============================] - 4s 383us/step - loss: 2.4013 - acc: 0.2432 - val_loss: 2.6009 - val_acc: 0.1985
    
    Epoch 00007: val_acc did not improve from 0.20700
    Epoch 8/200
    10000/10000 [==============================] - 4s 384us/step - loss: 2.3177 - acc: 0.2729 - val_loss: 2.9851 - val_acc: 0.1650
    
    Epoch 00008: val_acc did not improve from 0.20700
    Epoch 9/200
    10000/10000 [==============================] - 4s 384us/step - loss: 2.2545 - acc: 0.2941 - val_loss: 2.6890 - val_acc: 0.1808
    
    Epoch 00009: val_acc did not improve from 0.20700
    Epoch 10/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.1861 - acc: 0.3154 - val_loss: 2.3820 - val_acc: 0.2598
    
    Epoch 00010: val_acc improved from 0.20700 to 0.25983, saving model to step1_stage2_afterdylr.h5
    Epoch 11/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.1209 - acc: 0.3280 - val_loss: 2.3808 - val_acc: 0.2633
    
    Epoch 00011: val_acc improved from 0.25983 to 0.26333, saving model to step1_stage2_afterdylr.h5
    Epoch 12/200
    10000/10000 [==============================] - 4s 382us/step - loss: 2.0628 - acc: 0.3434 - val_loss: 2.2211 - val_acc: 0.2950
    
    Epoch 00012: val_acc improved from 0.26333 to 0.29500, saving model to step1_stage2_afterdylr.h5
    Epoch 13/200
    10000/10000 [==============================] - 4s 381us/step - loss: 2.0076 - acc: 0.3656 - val_loss: 3.1630 - val_acc: 0.1882
    
    Epoch 00013: val_acc did not improve from 0.29500
    Epoch 14/200
    10000/10000 [==============================] - 4s 380us/step - loss: 1.9456 - acc: 0.3853 - val_loss: 2.3239 - val_acc: 0.2552
    
    Epoch 00014: val_acc did not improve from 0.29500
    Epoch 15/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.8662 - acc: 0.4025 - val_loss: 2.5213 - val_acc: 0.2515
    
    Epoch 00015: val_acc did not improve from 0.29500
    Epoch 16/200
    10000/10000 [==============================] - 4s 380us/step - loss: 1.8106 - acc: 0.4188 - val_loss: 2.3896 - val_acc: 0.2820
    
    Epoch 00016: val_acc did not improve from 0.29500
    Epoch 17/200
    10000/10000 [==============================] - 4s 382us/step - loss: 1.7465 - acc: 0.4421 - val_loss: 2.2911 - val_acc: 0.2977
    
    Epoch 00017: val_acc improved from 0.29500 to 0.29767, saving model to step1_stage2_afterdylr.h5
    Epoch 18/200
    10000/10000 [==============================] - 4s 384us/step - loss: 1.6756 - acc: 0.4658 - val_loss: 2.3068 - val_acc: 0.3287
    
    Epoch 00018: val_acc improved from 0.29767 to 0.32867, saving model to step1_stage2_afterdylr.h5
    Epoch 19/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.6193 - acc: 0.4767 - val_loss: 2.1083 - val_acc: 0.3632
    
    Epoch 00019: val_acc improved from 0.32867 to 0.36317, saving model to step1_stage2_afterdylr.h5
    Epoch 20/200
    10000/10000 [==============================] - 4s 388us/step - loss: 1.5467 - acc: 0.4975 - val_loss: 2.0584 - val_acc: 0.3748
    
    Epoch 00020: val_acc improved from 0.36317 to 0.37483, saving model to step1_stage2_afterdylr.h5
    Epoch 21/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.4946 - acc: 0.5145 - val_loss: 2.0203 - val_acc: 0.3897
    
    Epoch 00021: val_acc improved from 0.37483 to 0.38967, saving model to step1_stage2_afterdylr.h5
    Epoch 22/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.4388 - acc: 0.5351 - val_loss: 2.0287 - val_acc: 0.3763
    
    Epoch 00022: val_acc did not improve from 0.38967
    Epoch 23/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.3590 - acc: 0.5549 - val_loss: 2.0295 - val_acc: 0.3858
    
    Epoch 00023: val_acc did not improve from 0.38967
    Epoch 24/200
    10000/10000 [==============================] - 4s 382us/step - loss: 1.3101 - acc: 0.5695 - val_loss: 2.2502 - val_acc: 0.3762
    
    Epoch 00024: val_acc did not improve from 0.38967
    Epoch 25/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.2660 - acc: 0.5817 - val_loss: 2.0941 - val_acc: 0.3922
    
    Epoch 00025: val_acc improved from 0.38967 to 0.39217, saving model to step1_stage2_afterdylr.h5
    Epoch 26/200
    10000/10000 [==============================] - 4s 383us/step - loss: 1.2168 - acc: 0.6049 - val_loss: 2.3382 - val_acc: 0.3795
    
    Epoch 00026: val_acc did not improve from 0.39217
    Epoch 27/200
    10000/10000 [==============================] - 4s 383us/step - loss: 1.1798 - acc: 0.6194 - val_loss: 2.4515 - val_acc: 0.3835
    
    Epoch 00027: val_acc did not improve from 0.39217
    Epoch 28/200
    10000/10000 [==============================] - 4s 384us/step - loss: 1.1108 - acc: 0.6331 - val_loss: 2.4492 - val_acc: 0.3958
    
    Epoch 00028: val_acc improved from 0.39217 to 0.39583, saving model to step1_stage2_afterdylr.h5
    Epoch 29/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.0442 - acc: 0.6493 - val_loss: 1.8684 - val_acc: 0.4565
    
    Epoch 00029: val_acc improved from 0.39583 to 0.45650, saving model to step1_stage2_afterdylr.h5
    Epoch 30/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.9763 - acc: 0.6753 - val_loss: 2.1511 - val_acc: 0.4317
    
    Epoch 00030: val_acc did not improve from 0.45650
    Epoch 31/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.9358 - acc: 0.6863 - val_loss: 2.1275 - val_acc: 0.4270
    
    Epoch 00031: val_acc did not improve from 0.45650
    Epoch 32/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.8771 - acc: 0.7084 - val_loss: 2.7799 - val_acc: 0.3722
    
    Epoch 00032: val_acc did not improve from 0.45650
    Epoch 33/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.8361 - acc: 0.7167 - val_loss: 2.4834 - val_acc: 0.4067
    
    Epoch 00033: val_acc did not improve from 0.45650
    Epoch 34/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.7987 - acc: 0.7291 - val_loss: 3.1042 - val_acc: 0.3450
    
    Epoch 00034: val_acc did not improve from 0.45650
    Epoch 35/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.7455 - acc: 0.7533 - val_loss: 2.1591 - val_acc: 0.4680
    
    Epoch 00035: val_acc improved from 0.45650 to 0.46800, saving model to step1_stage2_afterdylr.h5
    Epoch 36/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.7127 - acc: 0.7616 - val_loss: 2.6173 - val_acc: 0.4087
    
    Epoch 00036: val_acc did not improve from 0.46800
    Epoch 37/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.6988 - acc: 0.7661 - val_loss: 2.6864 - val_acc: 0.4017
    
    Epoch 00037: val_acc did not improve from 0.46800
    Epoch 38/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.6586 - acc: 0.7775 - val_loss: 3.5135 - val_acc: 0.3913
    
    Epoch 00038: val_acc did not improve from 0.46800
    Epoch 39/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.6119 - acc: 0.7937 - val_loss: 2.2821 - val_acc: 0.4692
    
    Epoch 00039: val_acc improved from 0.46800 to 0.46917, saving model to step1_stage2_afterdylr.h5
    Epoch 40/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.5810 - acc: 0.8060 - val_loss: 2.7925 - val_acc: 0.4322
    
    Epoch 00040: val_acc did not improve from 0.46917
    Epoch 41/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.5482 - acc: 0.8171 - val_loss: 3.0198 - val_acc: 0.3973
    
    Epoch 00041: val_acc did not improve from 0.46917
    Epoch 42/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.4201 - acc: 0.8590 - val_loss: 1.8944 - val_acc: 0.5298
    
    Epoch 00042: val_acc improved from 0.46917 to 0.52983, saving model to step1_stage2_afterdylr.h5
    Epoch 43/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.3112 - acc: 0.8957 - val_loss: 1.9044 - val_acc: 0.5365
    
    Epoch 00043: val_acc improved from 0.52983 to 0.53650, saving model to step1_stage2_afterdylr.h5
    Epoch 44/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.2806 - acc: 0.9064 - val_loss: 1.9136 - val_acc: 0.5455
    
    Epoch 00044: val_acc improved from 0.53650 to 0.54550, saving model to step1_stage2_afterdylr.h5
    Epoch 45/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.2645 - acc: 0.9098 - val_loss: 1.9052 - val_acc: 0.5528
    
    Epoch 00045: val_acc improved from 0.54550 to 0.55283, saving model to step1_stage2_afterdylr.h5
    Epoch 46/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.2476 - acc: 0.9140 - val_loss: 1.9961 - val_acc: 0.5483
    
    Epoch 00046: val_acc did not improve from 0.55283
    Epoch 47/200
    10000/10000 [==============================] - 4s 380us/step - loss: 0.2280 - acc: 0.9218 - val_loss: 1.9290 - val_acc: 0.5557
    
    Epoch 00047: val_acc improved from 0.55283 to 0.55567, saving model to step1_stage2_afterdylr.h5
    Epoch 48/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.2190 - acc: 0.9255 - val_loss: 1.9858 - val_acc: 0.5517
    
    Epoch 00048: val_acc did not improve from 0.55567
    Epoch 49/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.2140 - acc: 0.9265 - val_loss: 2.0167 - val_acc: 0.5497
    
    Epoch 00049: val_acc did not improve from 0.55567
    Epoch 50/200
    10000/10000 [==============================] - 4s 380us/step - loss: 0.2030 - acc: 0.9289 - val_loss: 2.0792 - val_acc: 0.5432
    
    Epoch 00050: val_acc did not improve from 0.55567
    Epoch 51/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1985 - acc: 0.9331 - val_loss: 2.0505 - val_acc: 0.5482
    
    Epoch 00051: val_acc did not improve from 0.55567
    Epoch 52/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1840 - acc: 0.9389 - val_loss: 2.1351 - val_acc: 0.5510
    
    Epoch 00052: val_acc did not improve from 0.55567
    Epoch 53/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1791 - acc: 0.9372 - val_loss: 2.0804 - val_acc: 0.5503
    
    Epoch 00053: val_acc did not improve from 0.55567
    Epoch 54/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1765 - acc: 0.9372 - val_loss: 2.1769 - val_acc: 0.5458
    
    Epoch 00054: val_acc did not improve from 0.55567
    Epoch 55/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1698 - acc: 0.9415 - val_loss: 2.1611 - val_acc: 0.5517
    
    Epoch 00055: val_acc did not improve from 0.55567
    Epoch 56/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1637 - acc: 0.9453 - val_loss: 2.1915 - val_acc: 0.5485
    
    Epoch 00056: val_acc did not improve from 0.55567
    Epoch 57/200
    10000/10000 [==============================] - 4s 386us/step - loss: 0.1526 - acc: 0.9528 - val_loss: 2.2654 - val_acc: 0.5467
    
    Epoch 00057: val_acc did not improve from 0.55567
    Epoch 58/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1594 - acc: 0.9441 - val_loss: 2.2223 - val_acc: 0.5467
    
    Epoch 00058: val_acc did not improve from 0.55567
    Epoch 59/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.1492 - acc: 0.9484 - val_loss: 2.3157 - val_acc: 0.5485
    
    Epoch 00059: val_acc did not improve from 0.55567
    Epoch 60/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.1484 - acc: 0.9508 - val_loss: 2.2566 - val_acc: 0.5513
    
    Epoch 00060: val_acc did not improve from 0.55567
    Epoch 61/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1394 - acc: 0.9535 - val_loss: 2.3403 - val_acc: 0.5427
    
    Epoch 00061: val_acc did not improve from 0.55567
    Epoch 62/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.1449 - acc: 0.9498 - val_loss: 2.3891 - val_acc: 0.5438
    
    Epoch 00062: val_acc did not improve from 0.55567
    Epoch 63/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1304 - acc: 0.9568 - val_loss: 2.3392 - val_acc: 0.5500
    
    Epoch 00063: val_acc did not improve from 0.55567
    Epoch 64/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1303 - acc: 0.9565 - val_loss: 2.3996 - val_acc: 0.5407
    
    Epoch 00064: val_acc did not improve from 0.55567
    Epoch 65/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1184 - acc: 0.9630 - val_loss: 2.3252 - val_acc: 0.5568
    
    Epoch 00065: val_acc improved from 0.55567 to 0.55683, saving model to step1_stage2_afterdylr.h5
    Epoch 66/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.1255 - acc: 0.9584 - val_loss: 2.5104 - val_acc: 0.5445
    
    Epoch 00066: val_acc did not improve from 0.55683
    Epoch 67/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1175 - acc: 0.9600 - val_loss: 2.3921 - val_acc: 0.5523
    
    Epoch 00067: val_acc did not improve from 0.55683
    Epoch 68/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1144 - acc: 0.9587 - val_loss: 2.3568 - val_acc: 0.5558
    
    Epoch 00068: val_acc did not improve from 0.55683
    Epoch 69/200
    10000/10000 [==============================] - 4s 380us/step - loss: 0.1034 - acc: 0.9647 - val_loss: 2.3947 - val_acc: 0.5597
    
    Epoch 00069: val_acc improved from 0.55683 to 0.55967, saving model to step1_stage2_afterdylr.h5
    Epoch 70/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1087 - acc: 0.9657 - val_loss: 2.4919 - val_acc: 0.5487
    
    Epoch 00070: val_acc did not improve from 0.55967
    Epoch 71/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1003 - acc: 0.9671 - val_loss: 2.5816 - val_acc: 0.5418
    
    Epoch 00071: val_acc did not improve from 0.55967
    Epoch 72/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0969 - acc: 0.9670 - val_loss: 2.5321 - val_acc: 0.5507
    
    Epoch 00072: val_acc did not improve from 0.55967
    Epoch 73/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0954 - acc: 0.9698 - val_loss: 2.4873 - val_acc: 0.5548
    
    Epoch 00073: val_acc did not improve from 0.55967
    Epoch 74/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0962 - acc: 0.9698 - val_loss: 2.4605 - val_acc: 0.5573
    
    Epoch 00074: val_acc did not improve from 0.55967
    Epoch 75/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0955 - acc: 0.9700 - val_loss: 2.4622 - val_acc: 0.5542
    
    Epoch 00075: val_acc did not improve from 0.55967
    Epoch 76/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0900 - acc: 0.9713 - val_loss: 2.4683 - val_acc: 0.5557
    
    Epoch 00076: val_acc did not improve from 0.55967
    Epoch 77/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.0885 - acc: 0.9717 - val_loss: 2.4596 - val_acc: 0.5555
    
    Epoch 00077: val_acc did not improve from 0.55967
    Epoch 78/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0890 - acc: 0.9699 - val_loss: 2.4587 - val_acc: 0.5562
    
    Epoch 00078: val_acc did not improve from 0.55967
    Epoch 79/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0853 - acc: 0.9741 - val_loss: 2.4598 - val_acc: 0.5583
    
    Epoch 00079: val_acc did not improve from 0.55967
    Epoch 80/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0847 - acc: 0.9723 - val_loss: 2.4708 - val_acc: 0.5583
    
    Epoch 00080: val_acc did not improve from 0.55967
    Epoch 81/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0793 - acc: 0.9755 - val_loss: 2.4808 - val_acc: 0.5610
    
    Epoch 00081: val_acc improved from 0.55967 to 0.56100, saving model to step1_stage2_afterdylr.h5
    Epoch 82/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0873 - acc: 0.9707 - val_loss: 2.4835 - val_acc: 0.5597
    
    Epoch 00082: val_acc did not improve from 0.56100
    Epoch 83/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0860 - acc: 0.9716 - val_loss: 2.5002 - val_acc: 0.5562
    
    Epoch 00083: val_acc did not improve from 0.56100
    Epoch 84/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0881 - acc: 0.9690 - val_loss: 2.4931 - val_acc: 0.5565
    
    Epoch 00084: val_acc did not improve from 0.56100
    Epoch 85/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0826 - acc: 0.9740 - val_loss: 2.4776 - val_acc: 0.5578
    
    Epoch 00085: val_acc did not improve from 0.56100
    Epoch 86/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0873 - acc: 0.9711 - val_loss: 2.4798 - val_acc: 0.5573
    
    Epoch 00086: val_acc did not improve from 0.56100
    Epoch 87/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0806 - acc: 0.9745 - val_loss: 2.4889 - val_acc: 0.5598
    
    Epoch 00087: val_acc did not improve from 0.56100
    Epoch 88/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0852 - acc: 0.9726 - val_loss: 2.4997 - val_acc: 0.5582
    
    Epoch 00088: val_acc did not improve from 0.56100
    Epoch 89/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0905 - acc: 0.9713 - val_loss: 2.5072 - val_acc: 0.5588
    
    Epoch 00089: val_acc did not improve from 0.56100
    Epoch 90/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0889 - acc: 0.9730 - val_loss: 2.4906 - val_acc: 0.5597
    
    Epoch 00090: val_acc did not improve from 0.56100
    Epoch 91/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0835 - acc: 0.9730 - val_loss: 2.4996 - val_acc: 0.5585
    
    Epoch 00091: val_acc did not improve from 0.56100
    Epoch 92/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0830 - acc: 0.9722 - val_loss: 2.5112 - val_acc: 0.5570
    
    Epoch 00092: val_acc did not improve from 0.56100
    Epoch 93/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0890 - acc: 0.9720 - val_loss: 2.5230 - val_acc: 0.5563
    
    Epoch 00093: val_acc did not improve from 0.56100
    Epoch 94/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0841 - acc: 0.9729 - val_loss: 2.5286 - val_acc: 0.5553
    
    Epoch 00094: val_acc did not improve from 0.56100
    Epoch 95/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.0816 - acc: 0.9743 - val_loss: 2.5245 - val_acc: 0.5538
    
    Epoch 00095: val_acc did not improve from 0.56100
    Epoch 96/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.0820 - acc: 0.9721 - val_loss: 2.5114 - val_acc: 0.5562
    
    Epoch 00096: val_acc did not improve from 0.56100
    Epoch 97/200
    10000/10000 [==============================] - 4s 390us/step - loss: 0.0821 - acc: 0.9731 - val_loss: 2.5055 - val_acc: 0.5568
    
    Epoch 00097: val_acc did not improve from 0.56100
    Epoch 98/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.0812 - acc: 0.9732 - val_loss: 2.5129 - val_acc: 0.5578
    
    Epoch 00098: val_acc did not improve from 0.56100
    Epoch 99/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0763 - acc: 0.9760 - val_loss: 2.5284 - val_acc: 0.5592
    
    Epoch 00099: val_acc did not improve from 0.56100
    Epoch 100/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0805 - acc: 0.9719 - val_loss: 2.5395 - val_acc: 0.5597
    
    Epoch 00100: val_acc did not improve from 0.56100
    Epoch 101/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0783 - acc: 0.9758 - val_loss: 2.5294 - val_acc: 0.5603
    
    Epoch 00101: val_acc did not improve from 0.56100



```python
test_model.load_weights('step1_stage2_afterdylr.h5')
train_scores = test_model.evaluate(x_train_small, y_train_small, verbose=1)
val_scores = test_model.evaluate(x_val, y_val, verbose=1)

print('Training loss:', train_scores[0],', training accuracy: ',train_scores[1])
print('Validation loss:', val_scores[0],', validation accuracy: ',val_scores[1])

real_epochs = len(history.history['acc'])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.arange(1,real_epochs+1,1),history.history['acc'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_acc'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend()

# Note: loss is the loss function that is optimized for multi-class classification
# i.e. the multi-class version of cross-entropy error
plt.subplot(1,2,2)
plt.plot(np.arange(1,real_epochs+1,1),history.history['loss'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_loss'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

    10000/10000 [==============================] - 3s 265us/step
    6000/6000 [==============================] - 2s 266us/step
    Training loss: 0.0119687244844 , training accuracy:  0.9986
    Validation loss: 2.48075385952 , validation accuracy:  0.561



![png](output_14_1.png)


The model is powerfull enough.
The curve is smooth after epoch 40.
Overfitting.


```python
model=load_model('step1_stage2_afterdylr.h5')
y_val_pred = model.predict(x_val)
val_predicted_class = np.argmax(y_val_pred, axis=1)

val_cm = confusion_matrix(np.argmax(y_val, axis=1), val_predicted_class)

cm = pd.DataFrame(val_cm, range(20), range(20))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
```


![png](output_16_0.png)


High missclassification between types of animals (8,11,12,16), insects (7) and invertebrates (13), vehicle 1 (18) and vehicle 2 (19), food containers (3) and household electrical devices (5).

## Model with Dynamic Learning Rate + Mix Max & Avg Pooling (Step1 - Stage2)


```python
def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 40:
        lrate = 0.00005
    if epoch > 70:
        lrate = 0.000005
    return lrate

test_model = Sequential()
#stack 1
test_model.add(Conv2D(96, (5, 5), padding='same',
                 input_shape=x_train.shape[1:], kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(96, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.1))

#stack2
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(128, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

#stack3
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(Conv2D(256, (5, 5), padding='same', kernel_constraint=max_norm(2.)))
test_model.add(BatchNormalization())
test_model.add(Activation('relu'))
test_model.add(AveragePooling2D(pool_size=(3, 3), strides = 2))
test_model.add(Dropout(0.2))

test_model.add(Flatten())
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))
test_model.add(Dense(1024, kernel_constraint=max_norm(2.)))
test_model.add(Activation('relu'))
test_model.add(Dropout(0.5))

#output
test_model.add(Dense(num_classes))
test_model.add(Activation('softmax'))

test_model.summary()

epochs = 200
batch_size = 512

Adam = keras.optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
test_model.compile(loss='categorical_crossentropy',
              optimizer= Adam,
              metrics=['accuracy'])
filepath = 'step1_stage2_avrpooling.h5'

callbacks = [EarlyStopping(monitor='val_acc', patience=20),
             ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True), 
             LearningRateScheduler(lr_schedule)]

print('Not using data augmentation!')
history = test_model.fit(x_train_small, y_train_small,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          shuffle=True, callbacks = callbacks)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_79 (Conv2D)           (None, 32, 32, 96)        7296      
    _________________________________________________________________
    batch_normalization_79 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_117 (Activation)  (None, 32, 32, 96)        0         
    _________________________________________________________________
    conv2d_80 (Conv2D)           (None, 32, 32, 96)        230496    
    _________________________________________________________________
    batch_normalization_80 (Batc (None, 32, 32, 96)        384       
    _________________________________________________________________
    activation_118 (Activation)  (None, 32, 32, 96)        0         
    _________________________________________________________________
    max_pooling2d_38 (MaxPooling (None, 15, 15, 96)        0         
    _________________________________________________________________
    dropout_52 (Dropout)         (None, 15, 15, 96)        0         
    _________________________________________________________________
    conv2d_81 (Conv2D)           (None, 15, 15, 128)       307328    
    _________________________________________________________________
    batch_normalization_81 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_119 (Activation)  (None, 15, 15, 128)       0         
    _________________________________________________________________
    conv2d_82 (Conv2D)           (None, 15, 15, 128)       409728    
    _________________________________________________________________
    batch_normalization_82 (Batc (None, 15, 15, 128)       512       
    _________________________________________________________________
    activation_120 (Activation)  (None, 15, 15, 128)       0         
    _________________________________________________________________
    max_pooling2d_39 (MaxPooling (None, 7, 7, 128)         0         
    _________________________________________________________________
    dropout_53 (Dropout)         (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_83 (Conv2D)           (None, 7, 7, 256)         819456    
    _________________________________________________________________
    batch_normalization_83 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_121 (Activation)  (None, 7, 7, 256)         0         
    _________________________________________________________________
    conv2d_84 (Conv2D)           (None, 7, 7, 256)         1638656   
    _________________________________________________________________
    batch_normalization_84 (Batc (None, 7, 7, 256)         1024      
    _________________________________________________________________
    activation_122 (Activation)  (None, 7, 7, 256)         0         
    _________________________________________________________________
    average_pooling2d_3 (Average (None, 3, 3, 256)         0         
    _________________________________________________________________
    dropout_54 (Dropout)         (None, 3, 3, 256)         0         
    _________________________________________________________________
    flatten_14 (Flatten)         (None, 2304)              0         
    _________________________________________________________________
    dense_40 (Dense)             (None, 1024)              2360320   
    _________________________________________________________________
    activation_123 (Activation)  (None, 1024)              0         
    _________________________________________________________________
    dropout_55 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_41 (Dense)             (None, 1024)              1049600   
    _________________________________________________________________
    activation_124 (Activation)  (None, 1024)              0         
    _________________________________________________________________
    dropout_56 (Dropout)         (None, 1024)              0         
    _________________________________________________________________
    dense_42 (Dense)             (None, 20)                20500     
    _________________________________________________________________
    activation_125 (Activation)  (None, 20)                0         
    =================================================================
    Total params: 6,847,220
    Trainable params: 6,845,300
    Non-trainable params: 1,920
    _________________________________________________________________
    Not using data augmentation!
    Train on 10000 samples, validate on 6000 samples
    Epoch 1/200
    10000/10000 [==============================] - 17s 2ms/step - loss: 2.9019 - acc: 0.1145 - val_loss: 3.0819 - val_acc: 0.1157
    
    Epoch 00001: val_acc improved from -inf to 0.11567, saving model to step1_stage2_avrpooling.h5
    Epoch 2/200
    10000/10000 [==============================] - 4s 384us/step - loss: 2.5993 - acc: 0.1954 - val_loss: 2.9962 - val_acc: 0.1505
    
    Epoch 00002: val_acc improved from 0.11567 to 0.15050, saving model to step1_stage2_avrpooling.h5
    Epoch 3/200
    10000/10000 [==============================] - 4s 383us/step - loss: 2.4346 - acc: 0.2376 - val_loss: 2.5837 - val_acc: 0.2052
    
    Epoch 00003: val_acc improved from 0.15050 to 0.20517, saving model to step1_stage2_avrpooling.h5
    Epoch 4/200
    10000/10000 [==============================] - 4s 383us/step - loss: 2.3415 - acc: 0.2659 - val_loss: 2.4255 - val_acc: 0.2565
    
    Epoch 00004: val_acc improved from 0.20517 to 0.25650, saving model to step1_stage2_avrpooling.h5
    Epoch 5/200
    10000/10000 [==============================] - 4s 384us/step - loss: 2.2273 - acc: 0.3020 - val_loss: 2.4564 - val_acc: 0.2463
    
    Epoch 00005: val_acc did not improve from 0.25650
    Epoch 6/200
    10000/10000 [==============================] - 4s 386us/step - loss: 2.1176 - acc: 0.3326 - val_loss: 2.2370 - val_acc: 0.3000
    
    Epoch 00006: val_acc improved from 0.25650 to 0.30000, saving model to step1_stage2_avrpooling.h5
    Epoch 7/200
    10000/10000 [==============================] - 4s 386us/step - loss: 2.0267 - acc: 0.3678 - val_loss: 2.4687 - val_acc: 0.2525
    
    Epoch 00007: val_acc did not improve from 0.30000
    Epoch 8/200
    10000/10000 [==============================] - 4s 387us/step - loss: 1.9656 - acc: 0.3866 - val_loss: 2.2362 - val_acc: 0.3122
    
    Epoch 00008: val_acc improved from 0.30000 to 0.31217, saving model to step1_stage2_avrpooling.h5
    Epoch 9/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.8522 - acc: 0.4186 - val_loss: 2.7912 - val_acc: 0.2365
    
    Epoch 00009: val_acc did not improve from 0.31217
    Epoch 10/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.7816 - acc: 0.4347 - val_loss: 2.3279 - val_acc: 0.3222
    
    Epoch 00010: val_acc improved from 0.31217 to 0.32217, saving model to step1_stage2_avrpooling.h5
    Epoch 11/200
    10000/10000 [==============================] - 4s 386us/step - loss: 1.6895 - acc: 0.4643 - val_loss: 1.9381 - val_acc: 0.4048
    
    Epoch 00011: val_acc improved from 0.32217 to 0.40483, saving model to step1_stage2_avrpooling.h5
    Epoch 12/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.6008 - acc: 0.4915 - val_loss: 2.1355 - val_acc: 0.3538
    
    Epoch 00012: val_acc did not improve from 0.40483
    Epoch 13/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.5328 - acc: 0.5087 - val_loss: 2.2042 - val_acc: 0.3755
    
    Epoch 00013: val_acc did not improve from 0.40483
    Epoch 14/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.4530 - acc: 0.5279 - val_loss: 1.9067 - val_acc: 0.4198
    
    Epoch 00014: val_acc improved from 0.40483 to 0.41983, saving model to step1_stage2_avrpooling.h5
    Epoch 15/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.3598 - acc: 0.5628 - val_loss: 2.0488 - val_acc: 0.3942
    
    Epoch 00015: val_acc did not improve from 0.41983
    Epoch 16/200
    10000/10000 [==============================] - 4s 384us/step - loss: 1.3245 - acc: 0.5750 - val_loss: 2.3202 - val_acc: 0.3752
    
    Epoch 00016: val_acc did not improve from 0.41983
    Epoch 17/200
    10000/10000 [==============================] - 4s 385us/step - loss: 1.2601 - acc: 0.5911 - val_loss: 2.0973 - val_acc: 0.4152
    
    Epoch 00017: val_acc did not improve from 0.41983
    Epoch 18/200
    10000/10000 [==============================] - 4s 382us/step - loss: 1.1961 - acc: 0.6120 - val_loss: 2.2164 - val_acc: 0.3878
    
    Epoch 00018: val_acc did not improve from 0.41983
    Epoch 19/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.1093 - acc: 0.6382 - val_loss: 2.0183 - val_acc: 0.4287
    
    Epoch 00019: val_acc improved from 0.41983 to 0.42867, saving model to step1_stage2_avrpooling.h5
    Epoch 20/200
    10000/10000 [==============================] - 4s 381us/step - loss: 1.0399 - acc: 0.6612 - val_loss: 2.3698 - val_acc: 0.3858
    
    Epoch 00020: val_acc did not improve from 0.42867
    Epoch 21/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.9836 - acc: 0.6753 - val_loss: 2.7656 - val_acc: 0.3660
    
    Epoch 00021: val_acc did not improve from 0.42867
    Epoch 22/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.9472 - acc: 0.6873 - val_loss: 2.1506 - val_acc: 0.4427
    
    Epoch 00022: val_acc improved from 0.42867 to 0.44267, saving model to step1_stage2_avrpooling.h5
    Epoch 23/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.8859 - acc: 0.7068 - val_loss: 2.2791 - val_acc: 0.4280
    
    Epoch 00023: val_acc did not improve from 0.44267
    Epoch 24/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.8180 - acc: 0.7329 - val_loss: 2.4290 - val_acc: 0.4153
    
    Epoch 00024: val_acc did not improve from 0.44267
    Epoch 25/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.7926 - acc: 0.7378 - val_loss: 2.5645 - val_acc: 0.4177
    
    Epoch 00025: val_acc did not improve from 0.44267
    Epoch 26/200
    10000/10000 [==============================] - 4s 380us/step - loss: 0.7865 - acc: 0.7452 - val_loss: 2.1018 - val_acc: 0.4480
    
    Epoch 00026: val_acc improved from 0.44267 to 0.44800, saving model to step1_stage2_avrpooling.h5
    Epoch 27/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.6958 - acc: 0.7739 - val_loss: 2.5004 - val_acc: 0.4142
    
    Epoch 00027: val_acc did not improve from 0.44800
    Epoch 28/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.6568 - acc: 0.7824 - val_loss: 2.5295 - val_acc: 0.4553
    
    Epoch 00028: val_acc improved from 0.44800 to 0.45533, saving model to step1_stage2_avrpooling.h5
    Epoch 29/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.6097 - acc: 0.7945 - val_loss: 2.3620 - val_acc: 0.4695
    
    Epoch 00029: val_acc improved from 0.45533 to 0.46950, saving model to step1_stage2_avrpooling.h5
    Epoch 30/200
    10000/10000 [==============================] - 4s 387us/step - loss: 0.5667 - acc: 0.8105 - val_loss: 2.7176 - val_acc: 0.4452
    
    Epoch 00030: val_acc did not improve from 0.46950
    Epoch 31/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.5340 - acc: 0.8227 - val_loss: 2.7457 - val_acc: 0.4477
    
    Epoch 00031: val_acc did not improve from 0.46950
    Epoch 32/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.4779 - acc: 0.8414 - val_loss: 3.3744 - val_acc: 0.4252
    
    Epoch 00032: val_acc did not improve from 0.46950
    Epoch 33/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.4373 - acc: 0.8554 - val_loss: 2.6980 - val_acc: 0.4743
    
    Epoch 00033: val_acc improved from 0.46950 to 0.47433, saving model to step1_stage2_avrpooling.h5
    Epoch 34/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.4516 - acc: 0.8520 - val_loss: 2.4701 - val_acc: 0.4552
    
    Epoch 00034: val_acc did not improve from 0.47433
    Epoch 35/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.4133 - acc: 0.8677 - val_loss: 2.4845 - val_acc: 0.4808
    
    Epoch 00035: val_acc improved from 0.47433 to 0.48083, saving model to step1_stage2_avrpooling.h5
    Epoch 36/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.3603 - acc: 0.8831 - val_loss: 3.0794 - val_acc: 0.4538
    
    Epoch 00036: val_acc did not improve from 0.48083
    Epoch 37/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.3505 - acc: 0.8826 - val_loss: 3.8395 - val_acc: 0.3908
    
    Epoch 00037: val_acc did not improve from 0.48083
    Epoch 38/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.3351 - acc: 0.8913 - val_loss: 2.9119 - val_acc: 0.4518
    
    Epoch 00038: val_acc did not improve from 0.48083
    Epoch 39/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.3275 - acc: 0.8974 - val_loss: 2.8265 - val_acc: 0.4478
    
    Epoch 00039: val_acc did not improve from 0.48083
    Epoch 40/200
    10000/10000 [==============================] - 4s 384us/step - loss: 0.3227 - acc: 0.8955 - val_loss: 2.9700 - val_acc: 0.4447
    
    Epoch 00040: val_acc did not improve from 0.48083
    Epoch 41/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.3084 - acc: 0.9015 - val_loss: 2.7521 - val_acc: 0.4732
    
    Epoch 00041: val_acc did not improve from 0.48083
    Epoch 42/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1913 - acc: 0.9385 - val_loss: 2.3197 - val_acc: 0.5180
    
    Epoch 00042: val_acc improved from 0.48083 to 0.51800, saving model to step1_stage2_avrpooling.h5
    Epoch 43/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.1210 - acc: 0.9629 - val_loss: 2.1692 - val_acc: 0.5475
    
    Epoch 00043: val_acc improved from 0.51800 to 0.54750, saving model to step1_stage2_avrpooling.h5
    Epoch 44/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.1010 - acc: 0.9706 - val_loss: 2.1390 - val_acc: 0.5583
    
    Epoch 00044: val_acc improved from 0.54750 to 0.55833, saving model to step1_stage2_avrpooling.h5
    Epoch 45/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0918 - acc: 0.9733 - val_loss: 2.1718 - val_acc: 0.5603
    
    Epoch 00045: val_acc improved from 0.55833 to 0.56033, saving model to step1_stage2_avrpooling.h5
    Epoch 46/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0759 - acc: 0.9773 - val_loss: 2.2176 - val_acc: 0.5600
    
    Epoch 00046: val_acc did not improve from 0.56033
    Epoch 47/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0736 - acc: 0.9780 - val_loss: 2.2597 - val_acc: 0.5643
    
    Epoch 00047: val_acc improved from 0.56033 to 0.56433, saving model to step1_stage2_avrpooling.h5
    Epoch 48/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0658 - acc: 0.9814 - val_loss: 2.3095 - val_acc: 0.5642
    
    Epoch 00048: val_acc did not improve from 0.56433
    Epoch 49/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0621 - acc: 0.9820 - val_loss: 2.3329 - val_acc: 0.5650
    
    Epoch 00049: val_acc improved from 0.56433 to 0.56500, saving model to step1_stage2_avrpooling.h5
    Epoch 50/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.0595 - acc: 0.9811 - val_loss: 2.3668 - val_acc: 0.5653
    
    Epoch 00050: val_acc improved from 0.56500 to 0.56533, saving model to step1_stage2_avrpooling.h5
    Epoch 51/200
    10000/10000 [==============================] - 4s 386us/step - loss: 0.0586 - acc: 0.9834 - val_loss: 2.4089 - val_acc: 0.5665
    
    Epoch 00051: val_acc improved from 0.56533 to 0.56650, saving model to step1_stage2_avrpooling.h5
    Epoch 52/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0556 - acc: 0.9837 - val_loss: 2.3708 - val_acc: 0.5678
    
    Epoch 00052: val_acc improved from 0.56650 to 0.56783, saving model to step1_stage2_avrpooling.h5
    Epoch 53/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0508 - acc: 0.9852 - val_loss: 2.3952 - val_acc: 0.5710
    
    Epoch 00053: val_acc improved from 0.56783 to 0.57100, saving model to step1_stage2_avrpooling.h5
    Epoch 54/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0496 - acc: 0.9852 - val_loss: 2.4542 - val_acc: 0.5682
    
    Epoch 00054: val_acc did not improve from 0.57100
    Epoch 55/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0464 - acc: 0.9856 - val_loss: 2.4625 - val_acc: 0.5692
    
    Epoch 00055: val_acc did not improve from 0.57100
    Epoch 56/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0441 - acc: 0.9865 - val_loss: 2.4862 - val_acc: 0.5687
    
    Epoch 00056: val_acc did not improve from 0.57100
    Epoch 57/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0374 - acc: 0.9905 - val_loss: 2.5556 - val_acc: 0.5628
    
    Epoch 00057: val_acc did not improve from 0.57100
    Epoch 58/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0402 - acc: 0.9891 - val_loss: 2.5644 - val_acc: 0.5605
    
    Epoch 00058: val_acc did not improve from 0.57100
    Epoch 59/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0390 - acc: 0.9886 - val_loss: 2.5200 - val_acc: 0.5658
    
    Epoch 00059: val_acc did not improve from 0.57100
    Epoch 60/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0399 - acc: 0.9876 - val_loss: 2.5615 - val_acc: 0.5678
    
    Epoch 00060: val_acc did not improve from 0.57100
    Epoch 61/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0380 - acc: 0.9891 - val_loss: 2.5819 - val_acc: 0.5665
    
    Epoch 00061: val_acc did not improve from 0.57100
    Epoch 62/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0346 - acc: 0.9887 - val_loss: 2.6285 - val_acc: 0.5688
    
    Epoch 00062: val_acc did not improve from 0.57100
    Epoch 63/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0354 - acc: 0.9894 - val_loss: 2.6094 - val_acc: 0.5690
    
    Epoch 00063: val_acc did not improve from 0.57100
    Epoch 64/200
    10000/10000 [==============================] - 4s 381us/step - loss: 0.0328 - acc: 0.9894 - val_loss: 2.6595 - val_acc: 0.5657
    
    Epoch 00064: val_acc did not improve from 0.57100
    Epoch 65/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0343 - acc: 0.9893 - val_loss: 2.7738 - val_acc: 0.5600
    
    Epoch 00065: val_acc did not improve from 0.57100
    Epoch 66/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0325 - acc: 0.9891 - val_loss: 2.6615 - val_acc: 0.5690
    
    Epoch 00066: val_acc did not improve from 0.57100
    Epoch 67/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0301 - acc: 0.9912 - val_loss: 2.6608 - val_acc: 0.5663
    
    Epoch 00067: val_acc did not improve from 0.57100
    Epoch 68/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0309 - acc: 0.9913 - val_loss: 2.7794 - val_acc: 0.5625
    
    Epoch 00068: val_acc did not improve from 0.57100
    Epoch 69/200
    10000/10000 [==============================] - 4s 383us/step - loss: 0.0295 - acc: 0.9919 - val_loss: 2.7255 - val_acc: 0.5677
    
    Epoch 00069: val_acc did not improve from 0.57100
    Epoch 70/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0273 - acc: 0.9919 - val_loss: 2.8008 - val_acc: 0.5628
    
    Epoch 00070: val_acc did not improve from 0.57100
    Epoch 71/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0283 - acc: 0.9906 - val_loss: 2.8409 - val_acc: 0.5632
    
    Epoch 00071: val_acc did not improve from 0.57100
    Epoch 72/200
    10000/10000 [==============================] - 4s 385us/step - loss: 0.0304 - acc: 0.9910 - val_loss: 2.7947 - val_acc: 0.5643
    
    Epoch 00072: val_acc did not improve from 0.57100
    Epoch 73/200
    10000/10000 [==============================] - 4s 382us/step - loss: 0.0256 - acc: 0.9923 - val_loss: 2.7680 - val_acc: 0.5670
    
    Epoch 00073: val_acc did not improve from 0.57100



```python
test_model.load_weights('step1_stage2_avrpooling.h5')
train_scores = test_model.evaluate(x_train_small, y_train_small, verbose=1)
val_scores = test_model.evaluate(x_val, y_val, verbose=1)

print('Training loss:', train_scores[0],', training accuracy: ',train_scores[1])
print('Validation loss:', val_scores[0],', validation accuracy: ',val_scores[1])

real_epochs = len(history.history['acc'])
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(np.arange(1,real_epochs+1,1),history.history['acc'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_acc'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0,1])
plt.legend()

# Note: loss is the loss function that is optimized for multi-class classification
# i.e. the multi-class version of cross-entropy error
plt.subplot(1,2,2)
plt.plot(np.arange(1,real_epochs+1,1),history.history['loss'],'g-',label='training')
plt.plot(np.arange(1,real_epochs+1,1),history.history['val_loss'],'r-',label='validation')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()
```

    10000/10000 [==============================] - 3s 289us/step
    6000/6000 [==============================] - 2s 279us/step
    Training loss: 0.00740174557311 , training accuracy:  0.9988
    Validation loss: 2.39517573166 , validation accuracy:  0.571



![png](output_20_1.png)



```python
model=load_model('step1_stage2_avrpooling.h5')
y_val_pred = model.predict(x_val)
val_predicted_class = np.argmax(y_val_pred, axis=1)

val_cm = confusion_matrix(np.argmax(y_val, axis=1), val_predicted_class)

cm = pd.DataFrame(val_cm, range(20), range(20))
plt.figure(figsize = (20,14))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cm, annot=True, annot_kws={"size": 12}) # font size
plt.show()
```


![png](output_21_0.png)


Missclassification between types of animals (8,11,12,16) is reduced slightly.
Missclassification between insects (7) and invertebrates (13) is also decreased. 

With the mix between average pooling and max pooling, the val acc is slightly higher than the model with only maxpooling by 1%. 
