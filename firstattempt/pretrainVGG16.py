from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array

import numpy as np
from numpy import array
import glob
import os
import matplotlib.pyplot as plt
import cv2
import keras
import argparse
import random

from fourconlayer_model import  create_model_four_conv
from paths import list_images
import sklearn

from keras.applications import VGG16

#mz own classifer
#conv_base.summary()
#after model.compile, model.summary



ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset', required=True,
                help="path to input dataset")
#ap.add_argument('-i','--images', required=True,
#                help="path to test dataset")
ap.add_argument('-m','--model', required=True,
                help="path to input model")
ap.add_argument('-p','--plot',type=str,default="plot.png",
                help="path to output accuracy")
args=vars(ap.parse_args())

dict={0:"caudalie",1:"niveaenergy",2:"niveasport",3:"red_conditioner",4:"red_shampoo",5:"shampoo_apple",6:"shampoo_lemon"}



train_image_paths=list(list_images(args["dataset"])) #fit_generator
random.seed(42)
random.shuffle(train_image_paths)

def dataset(image_paths):
    data=[]
    labels=[]
    for imagepath in image_paths:
        #print("imagepath:",imagepath)
        image=cv2.imread(imagepath)
        image=cv2.resize(image,(150,150))
        image=img_to_array(image)
        data.append(image)

        label=imagepath.split(os.path.sep)[-2]
        for key,value in dict.items():
            if label==value:
                labels.append(key)
            else:
                pass
    return data,labels

data,labels=dataset(train_image_paths)

data=np.array(data,dtype="float")/255.0
labels=np.array(labels)

print("data size",len(data))
'''
test_image_paths=list(list_images(args["images"]))
test_data,test_labels=dataset(test_image_paths)
test_data=np.array(test_data,dtype="float")/255.0
test_labels=np.array(test_labels)
'''
#split data for test,val
(trainX,testX,trainY, testY)=train_test_split(data,labels,test_size=0.2,random_state=42)
print("traindata size",len(trainX))

num_classes=len(dict)
#print(num_classes)
trainY=to_categorical(trainY,num_classes)
testY=to_categorical(testY,num_classes)
#test_labels=to_categorical(test_labels,num_classes)

conv_base=VGG16(weights='imagenet',
include_top=False,
input_shape=(150, 150, 3))

datagen=ImageDataGenerator()
batch_size=3

def extract_feature(dataset,labels,nb_sample):
    features=np.zeros(shape=(nb_sample,4,4,512))
    #labels=np.zeros(shape=(nb_sample,num_classes))
    generator = datagen.flow(
    dataset,labels,batch_size=batch_size
    )
    i=0
    for input_batch,label_batch in generator:
        features_batch=conv_base.predict(input_batch)
        features[i*batch_size:(i+1)*batch_size]= features_batch
        labels[i*batch_size:(i+1)*batch_size]= label_batch
        i +=1
        if i*batch_size>=nb_sample:
            break
    return features,labels

train_features, train_labels = extract_feature(trainX,trainY, len(trainX))
print("lenth of train_features",len(train_features))
validation_features, validation_labels = extract_feature(testX,testY,len(testX))
#test_features,test_labels=extract_feature(test_data,test_labels,len(test_data))

train_features = np.reshape(train_features, (len(trainX), 4 * 4 * 512))
validation_features = np.reshape(validation_features, (len(testX), 4 * 4 * 512))
#test_features=np.reshape(test_features,(len(test_data),4*4*512))

model=models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-6),loss='categorical_crossentropy',metrics=['acc'])

history = model.fit(
train_features, train_labels,
epochs=80,
batch_size=3,
validation_data=(validation_features, validation_labels),
verbose=1)

model.save(args["model"])
'''
labels_predict=model.predict(test_features)

print('\n', sklearn.metrics.classification_report(np.where(test_labels > 0)[1],
                                                  np.argmax(labels_predict, axis=1),
                                                  target_names=list(dict.values())), sep='')
'''
