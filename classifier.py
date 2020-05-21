import numpy as np                                                      # linear algebra
import pandas as pd                                                     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                                         # Ploting charts
from glob import glob                                                   # retriving an array of files in directories

import os
import cv2
import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential                          # for neural network models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D ,ZeroPadding2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator     # Data augmentation and preprocessing
from tensorflow.keras.utils import to_categorical                       # For One-hot Encoding
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

def load_data():
    #CREATE TRAIN DATA AND LABELS
    try:
        print("-----------> Trying to load the training data:")
        train_x = np.load('data_1/train_x.npz')['train_x']
        train_y = np.load('data_1/train_y.npz')['train_y']
        print("-----------> Successfully loaded the data:")

    except:
        print("-----------> Failed to loaded the datas:")
        print("-----------> Starting to aggregating the data:")
        train_x = []
        train_y = []
        
        for root, dirs, files in os.walk('chest_xray/chest_xray/train'):
            if 'NORMAL' in root:
                target = 0
            else:
                target = 1
            for file in files:
                im = cv2.resize(cv2.imread(root + '/' + file), (224, 224)).astype(np.float32) / 255.0
                #Standardize each image separately
                #im = tf.image.per_image_standardization(im)
                train_x.append(im)
                train_y.append(target)
        np.savez('data_1/train_x.npz', train_x=train_x)
        np.savez('data_1/train_y.npz', train_y=train_y)



    #CREATE VALIDATION DATA AND LABELS
    '''try:
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
        np.savez('data/val_y.npz', val_y=val_y)'''

    #CREATE TEST DATA AND LABELS
    try:
        print("-----------> Trying to load the testing data:")
        test_x = np.load('data_1/test_x.npz')['test_x']
        test_y = np.load('data_1/test_y.npz')['test_y']
        print("-----------> Successfully loaded the data:")

    except:
        print("-----------> Failed to loaded the datas:")
        print("-----------> Starting to aggregating the data:")
        test_x = []
        test_y = []
        
        for root, dirs, files in os.walk('chest_xray/chest_xray/test'):
            if 'NORMAL' in root:
                target = 0
            else:
                target = 1
            for file in files:
                
                im = cv2.resize(cv2.imread(root + '/' + file), (224, 224)).astype(np.float32) / 255.0
                test_x.append(im)
                test_y.append(target)
        np.savez('data_1/test_x.npz', test_x=test_x)
        np.savez('data_1/test_y.npz', test_y=test_y)
    return train_x, train_y, test_x, test_y

def create_model():
    model = tf.keras.models.Sequential([

        ZeroPadding2D(padding = (1,1), input_shape = (226, 226, 3), trainable = False),
        Conv2D(64,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(64,(3,3), activation = 'relu', trainable = False),
        MaxPool2D(pool_size = (2,2), strides = (2,2), trainable = False),# pading = 'same'),
        #Dropout()

        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(128,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(128,(3,3), activation = 'relu', trainable = False),
        MaxPool2D(pool_size = (2,2), strides = (2,2), trainable = False), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1), trainable = False), 
        Conv2D(256,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(256,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(256,(3,3), activation = 'relu', trainable = False),
        MaxPool2D(pool_size = (2,2), strides = (2,2), trainable = False), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        MaxPool2D(pool_size = (2,2), strides = (2,2), trainable = False), #, padding = 'same'),

        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        ZeroPadding2D(padding = (1,1), trainable = False),
        Conv2D(512,(3,3), activation = 'relu', trainable = False),
        MaxPool2D(pool_size = (2,2), strides = (2,2), trainable = False), #, padding = 'same'),



        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.2),
        Dense(4096, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')

    ])
    if os.path.isfile('weights5_epoch'):
        model.load_weights('weights5_epoch')
        ##model.trainable = False
    #lr for pretraining: 0.0001
    #lr for fine-tuning: 0.00001
    model.compile(optimizer=Adam(lr=0.00001),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])#,early_stopping_monitor = EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2))
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
    train_x, train_y, test_x, test_y = load_data()
    #new_model = create_model()
    vgg16 = VGG16(weights='imagenet', include_top=True)

    #Add a layer where input is the output of the  second last layer 
    x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    #Then create the corresponding model 
    
    for layer in vgg16.layers:
        layer.trainable = False
    new_model = Model(inputs=vgg16.input, outputs=x)
    new_model.compile(optimizer=Adam(lr=0.0001),loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    print("-------------TEST COOL --------------")
    #Viewing the summary of the model
    covid_test_x = np.load('data_2/test_x.npz')['test_x']
    covid_test_y = np.load('data_2/test_y.npz')['test_y']
    covid_train_x = np.load('data_2/train_x.npz')['train_x']
    covid_train_y = np.load('data_2/train_y.npz')['train_y']


    #,train_x[-260:,:,:,:]
    #,train_y[-260:]
    combined_train_x = np.concatenate((covid_train_x,train_x[:260, :,:,:]),axis = 0)
    combined_train_y = np.concatenate((covid_train_y-1,train_y[:260]),axis = 0)
    combined_test_x = np.concatenate((covid_test_x,test_x[:327,:,:,:]),axis = 0)
    combined_test_y = np.concatenate((covid_test_y-1,test_y[:327]),axis = 0)
    rng_state = np.random.get_state()
    print(combined_train_x.shape)
    np.random.shuffle(combined_train_x)
    np.random.set_state(rng_state)
    np.random.shuffle(combined_train_y)

    history = new_model.fit(combined_train_x, combined_train_y, epochs=3, validation_split=0.125, shuffle=True, callbacks=[EarlyStopping(patience = 3, monitor = "val_acc", mode="max", verbose = 2)])
    #new_model.save('weights_2xcovid')
    #new_model.load_weights('weights_2xcovid')
    #model.load_weights('weights_2xcovid')
    #tf.keras.layers.Dropout(
    #    rate, noise_shape=None, seed=None, **kwargs
    #)

    # rate: Float between 0 and 1. Fraction of the input units to drop.

    # noise_shape: 1D integer tensor representing the shape of the binary 
    # dropout mask that will be multiplied with the input. For instance, 
    # if your inputs have shape (batch_size, timesteps, features) and you want 
    # the dropout mask to be the same for all timesteps, you can use noise_shape=(batch_size, 1, features).

    # seed: A Python#SkyllBaraAlltPåCOOOOORONA integer to use as random seed.

    #predictions = model.predict(test_x, verbose = 1)
    #print(np.argmax(predictions))
    #  "Accuracy"
    print(history.history['acc'])
    print(history.history['val_acc'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    #new_model.evaluate(combined_test_x,  combined_test_y, verbose=1)
    #new_model.evaluate(test_x[326:,:,:],  test_y[326:], verbose=1)

    #new_model.evaluate(test_x[:326],  test_y[:326], verbose=1)
    #new_model.evaluate(covid_test_x,  covid_test_y-1, verbose=1)
    #model.evaluate(covid_test_x,  covid_test_y, verbose=1)
    


    
if __name__ == "__main__":
    main()



