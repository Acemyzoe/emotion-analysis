#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Acemyzoe'

import os
import cv2
import numpy as np

def resize(pic_name):    
    cascPath = "./model/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(pic_name)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = image[y:y+h,x:x+w]
        image = cv2.resize(image,(48,48))
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('image',image)
        image_path = pic_name+'_resize.jpg'
        pic = cv2.imwrite(image_path,image)
    return image_path

if __name__=='__main__':
    path = resize('./data/obama.jpg')
    print(path)
    


