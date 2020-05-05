import numpy as np                                                      # linear algebra
import pandas as pd                                                     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                                         # Ploting charts
from glob import glob                                                   # retriving an array of files in directories

import os
import cv2
import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential                          # for neural network models
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D ,ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # Data augmentation and preprocessing
from tensorflow.keras.utils import to_categorical                       # For One-hot Encoding
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    #CREATE TRAIN DATA AND LABELS
    try:
        print("-----------> Trying to load the training data:")
        train_x = np.load('data/train_x.npz')['train_x']
        train_y = np.load('data/train_y.npz')['train_y']
        print("-----------> Successfully loaded the data:")

    except:
        print("-----------> Failed to loaded the datas:")
        print("-----------> Starting to aggregating the data:")
        train_x = []
        train_y = []
        
        for root, dirs, files in os.walk('chest_xray/train'):
            if 'NORMAL' in root:
                target = 0
            else:
                target = 1
            for file in files:
                
                im = cv2.resize(cv2.imread(root + '/' + file), (226, 226)).astype(np.float32)
                train_x.append(im)
                train_y.append(target)
        np.savez('data/train_x.npz', train_x=train_x)
        np.savez('data/train_y.npz', train_y=train_y)
    #CREATE VALIDATION DATA AND LABELS
    try:
        print("-----------> Trying to load the validation data:")
        val_x = np.load('data/val_x.npz')['val_x']
        val_y = np.load('data/val_y.npz')['val_y']
        print("-----------> Successfully loaded the data:")

    except:
        print("-----------> Failed to loaded the datas:")
        print("-----------> Starting to aggregating the data:")
        val_x = []
        val_y = []
        
        for root, dirs, files in os.walk('chest_xray/val'):
            if 'NORMAL' in root:
                target = 0
            else:
                target = 1
            for file in files:
                
                im = cv2.resize(cv2.imread(root + '/' + file), (226, 226)).astype(np.float32)
                val_x.append(im)
                val_y.append(target)
        np.savez('data/val_x.npz', val_x=val_x)
        np.savez('data/val_y.npz', val_y=val_y)

    #CREATE TEST DATA AND LABELS
    try:
        print("-----------> Trying to load the testing data:")
        test_x = np.load('data/test_x.npz')['test_x']
        test_y = np.load('data/test_y.npz')['test_y']
        print("-----------> Successfully loaded the data:")

    except:
        print("-----------> Failed to loaded the datas:")
        print("-----------> Starting to aggregating the data:")
        test_x = []
        test_y = []
        
        for root, dirs, files in os.walk('chest_xray/test'):
            if 'NORMAL' in root:
                target = 0
            else:
                target = 1
            for file in files:
                
                im = cv2.resize(cv2.imread(root + '/' + file), (226, 226)).astype(np.float32)
                test_x.append(im)
                test_y.append(target)
        np.savez('data/test_x.npz', test_x=test_x)
        np.savez('data/test_y.npz', test_y=test_y)
    return train_x, train_y, val_x, val_y, test_x, test_y

def create_model():
    model = tf.keras.models.Sequential([

        ZeroPadding2D(padding = (1,1), input_shape = (226, 226, 3)),
        Conv2D(64,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(64,(3,3), activation = 'relu'),
        MaxPool2D(pool_size = (2,2), strides = (2,2)),# pading = 'same'),


        ZeroPadding2D(padding = (1,1)),
        Conv2D(128,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(128,(3,3), activation = 'relu'),
        MaxPool2D(pool_size = (2,2), strides = (2,2)), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1)), 
        Conv2D(256,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(256,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(256,(3,3), activation = 'relu'),
        MaxPool2D(pool_size = (2,2), strides = (2,2)), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1)),
        Conv2D(512,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(512,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
         Conv2D(512,(3,3), activation = 'relu'),
        MaxPool2D(pool_size = (2,2), strides = (2,2)), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1)),
        Conv2D(512,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(512,(3,3), activation = 'relu'),
        ZeroPadding2D(padding = (1,1)),
        Conv2D(512,(3,3), activation = 'relu'),
        MaxPool2D(pool_size = (2,2), strides = (2,2)), #, padding = 'same'),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer=Adam(lr=0.0001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'],
                

                # early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)
                )
    return model

'''
im[:,:,0] -= 103.939
im[:,:,1] -= 116.779
im[:,:,2] -= 123.68
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)
out = model.predict(im)
print(np.argmax(out))'''
def main(): 
    train_x, train_y, val_x, val_y, test_x, test_y = load_data()
    print(train_x.shape)
    print(val_x.shape)
    print(test_x.shape)
    model = create_model()
    
    #Viewing the summary of the model
    model.summary()


    if os.path.isfile('weights'):
        model.load_weights('weights')
    else:
        model.fit(train_x, train_y, epochs=5, validation_data=(val_x, val_y), shuffle=True, callbacks=[EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)])
        model.save('weights')
    model.evaluate(val_x,  val_y, verbose=1)

    model.evaluate(test_x,  test_y, verbose=1)



if __name__ == "__main__":
    main()



