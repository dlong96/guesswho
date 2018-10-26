import numpy as np
import cv2

class CropPreprocessor:
    #target width and height
    def __init__(self,width,height,horizon=True,inter=cv2.INTER_AREA):
        self.width=width
        self.height=height
        self.horizon=horizon
        self.inter=inter

def preprocess(self,image):
    crops=[]

    (h,w)=iamge.shape[:2]
    #four corners  upper& lower
    coordinates=[
        [0,0,self.width,self.height],
        [w-self.width,0,w,self.height],
        [w-self.width,h-self.height,w,h]
        [0,h-self.height,self.width,h] ]
           ]
    dw= int(0.5*(w-self.width))
    dh= int(0.5*(h-self.height))
    coordinates.append([dw,dh,w-dw,h-dh])

    for (x1,y1,x2,y2) in coordinates:
        crop=image[y1:y2,x1:x2]
        crop=cv2.resize(crop,(self.width,self.height),interpolation=self.inter)
        crops.append(crop)

    if self.horizon:
        mirror_images=[cv2.flip(c,1) for c in crops]
        crops.extend(mirrors)

    return np.array(crops)

    
