from paths import list_images
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
from keras.models import load_model
import sklearn

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
from mpl_toolkits.axes_grid1 import AxesGrid

ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,
help="path to model")
ap.add_argument("-i","--images",required=True,
help="path to images")
ap.add_argument('-p','--plot',type=str,default="plot.png",
                help="path to output accuracy")

args=vars(ap.parse_args())

dict={0:"qizihan",1:"longdanni"}

X_test=[]
Y_test=[]
num_class=len(dict)

image_paths=list(list_images(args["images"])) #fit_generator
random.seed(42)
random.shuffle(image_paths)
copy=[]
for imagepath in image_paths:
    #print("imagepath:",imagepath)
    image=cv2.imread(imagepath)
    copy.append(image)
    image=cv2.resize(image,(128,128))
    image=img_to_array(image)
    X_test.append(image)

    label=imagepath.split(os.path.sep)[-2]
    #label=1 if label=="training_bottle" else 0
    for key,value in dict.items():
        if label==value:
            Y_test.append(key)

X_test=np.array(X_test,dtype="float")/255.0
Y_test=np.array(Y_test)
print("test site",len(X_test))
Y_test=to_categorical(Y_test,num_class)

print("[loadind network]")
model=load_model(args["model"])
Y_predict=model.predict(X_test)

print('\n', sklearn.metrics.classification_report(np.where(Y_test > 0)[1],
                                                  np.argmax(Y_predict, axis=1),
                                                  target_names=list(dict.values())), sep='')

plt.figure(figsize=(8,8))
cnf_matrix = sklearn.metrics.confusion_matrix(np.where(Y_test > 0)[1], np.argmax(Y_predict, axis=1))
classes = list(dict.values())
plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)

plt.show(1)

F = plt.figure(1, (15,20))
grid = AxesGrid(F, 111, nrows_ncols=(4, 4), axes_pad=0, label_mode="1")

for i,img in enumerate (copy[:16]):
    result=Y_predict[i]
    text =sorted(["{}:{:.2f}%".format(dict[k].title(),100*v) for k,v in enumerate(result)],key=lambda x:float(x.split(':')[1].split('%')[0]), reverse=True)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    output = cv2.resize(img, (300,300))
    for k, t in enumerate(text[:3]):
    	cv2.putText(output, t,(10, 30+k*20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)
    grid[i].imshow(output)


plt.draw()
plt.show(2)
plt.savefig(args["plot"])
