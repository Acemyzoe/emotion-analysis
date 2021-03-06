#!/usr/bin/python
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

import Face_extraction

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class cnn_model:
    def __init__(self):
        self.model = None

    def build_model(self):
        #construct CNN structure
        self.model = Sequential()
        #1st convolution layer
        self.model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
        self.model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

        #2nd convolution layer
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        #3rd convolution layer
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

        self.model.add(Flatten())

        #fully connected neural networks
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(num_classes, activation='softmax'))
        
        self.model.summary()

    def train_model(self):
        with open("./data/fer2013/fer2013.csv") as f:
            content = f.readlines()
        lines = np.array(content)
        num_of_instances = lines.size
        print("number of instances: ",num_of_instances)
        print("instance length: ",len(lines[1].split(",")[1].split(" ")))

        #init train set & test set
        x_train, y_train, x_test, y_test = [], [], [], []
        #transfer train and test set data
        for i in range(1,num_of_instances):
            try:
                emotion, img, usage = lines[i].split(",")
                val = img.split(" ")
                pixels = np.array(val, 'float32')
                emotion = keras.utils.to_categorical(emotion, num_classes)
    
                if 'Training' in usage:
                    y_train.append(emotion)
                    x_train.append(pixels)
                elif 'PublicTest' in usage:
                    y_test.append(emotion)
                    x_test.append(pixels)
            except:
	            print("error")

        #data transformation for train and test sets
        x_train = np.array(x_train, 'float32')
        y_train = np.array(y_train, 'float32')
        x_test = np.array(x_test, 'float32')
        y_test = np.array(y_test, 'float32')

        x_train /= 255 #normalize inputs between [0, 1]
        x_test /= 255

        x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
        x_train = x_train.astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
        x_test = x_test.astype('float32')

        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        #batch process
        gen = ImageDataGenerator()
        train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

        self.model.compile(loss='categorical_crossentropy'
            , optimizer=keras.optimizers.Adam()
            , metrics=['accuracy']
        )
        fit = self.model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)

    def save_model(self):
        #将模型结构序列化为JSON
        model_json = self.model.to_json()
        with open("./model/faceial_emotion_model_01.json","w") as json_file:
            json_file.write(model_json)
        #保存训练权重    
        self.model.save_weights('./model/facial_emotion_model_01.h5')

    def evaluation(self):
        score = self.model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', 100*score[1])   


def get_model():
    model = cnn_model()
    model.build_model()
    model.train_model()
    model.save_model()

def emotion_analysis(path):
    #载入model
    model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
    model.load_weights('./model/facial_expression_model_weights.h5')
    
    #载入picture
    img_resize = Face_extraction.resize(path)
    img = image.load_img(img_resize, color_mode="grayscale", target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    custom = model.predict(x)
    
    #draw chart
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))  
    plt.bar(y_pos, custom[0], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')    
    plt.show()



if __name__ == '__main__':

    #修改网络训练的参数
    num_classes = 7
    batch_size = 128
    epochs = 100

    get_model()

    #分析一张图片
    emotion_analysis('./data/cg1.jpg')

