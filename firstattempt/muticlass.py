from keras import models
from keras import layers
from keras import optimizers

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


ap=argparse.ArgumentParser()
ap.add_argument('-d','--dataset', required=True,
                help="path to input dataset")
ap.add_argument('-m','--model', required=True,
                help="path to input model")
ap.add_argument('-p','--plot',type=str,default="plot.png",
                help="path to output accuracy")

args=vars(ap.parse_args())



data=[]
labels=[]

#key,value  use item() show both
dict={0:"qizihan",1:"longdanni"}

#dict={0:"contonelle",1:"contonelle_blue",2:"cottonelle_box",3:"deluxe",4:"diaper3",5:"diaper4",6:"Kamili",
#7:"kuchentucher",8:"pad48",9:"pad58",10:"yellow_papier"}

image_paths=list(list_images(args["dataset"])) #fit_generator
random.seed(42)
random.shuffle(image_paths)

for imagepath in image_paths:
    #print("imagepath:",imagepath)
    image=cv2.imread(imagepath)
    image=cv2.resize(image,(128,128))
    image=img_to_array(image)
    data.append(image)

    label=imagepath.split(os.path.sep)[-2]
    #label=1 if label=="training_bottle" else 0
    for cls,name in dict.items():
        if label==name:
            labels.append(cls)
        else:
            pass

data=np.array(data,dtype="float")/255.0
labels=np.array(labels)
print("data site",len(data))
print("label",len(labels))

#split data for test,val
(trainX,testX,trainY, testY)=train_test_split(data,labels,test_size=0.2,random_state=42)
print("traindata size",len(trainX))

num_classes=len(dict)
trainY=to_categorical(trainY,num_classes=num_classes)
testY=to_categorical(testY,num_classes=num_classes)


model,opt= create_model_four_conv(num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    #rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

print('Training NN')
#batch_size=16
batch_size=12
train_generator = train_datagen.flow(trainX, trainY, batch_size=12)
#validation_generator = valtest_datagen.flow(testX, testY)
epochs=100

history = model.fit_generator(
    train_generator,
    validation_data=(testX,testY),
    steps_per_epoch= len(trainX) // batch_size,
    epochs=epochs,
    verbose=1
)
#verbose=0 silence =1 process bar =2 line per epooch
#model.save_weights('bin-class.h5')
model.save(args["model"])


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(epochs),history.history["loss"],"bo",label="training loss")
plt.plot(np.arange(epochs), history.history["val_loss"],label="val_loss")
plt.plot(np.arange(epochs),history.history["acc"],label="training acc")
plt.plot(np.arange(epochs), history.history["val_acc"],label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epochs")
plt.ylabel("loss/acc")
plt.legend(loc="lower left")
plt.show()
plt.savefig(args["plot"])
