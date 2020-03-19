#encoding:utf-8

'''
   csv to picture
'''

import pandas as pd
import numpy as np
import os
import cv2

emotions = {
    "0":"anger",
    "1":"disgust",
    "2":"fear",
    "3":"happy",
    "4":"sad",
    "5":"surprised",
    "6":"neutral"
}

def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)

        
def saveImageFromFer2013(file):
    
    # 读取csv文件
    faces_data = pd.read_csv(file)
    imageCount = 0
    # 遍历csv文件内容，并将图片数据按分类保存
    for index in range(len(faces_data)):
        # 解析每一行csv文件内容
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        # 将图片数据转换为48*48
        data_array = list(map(float,image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48,48)
        
        # 选择分类，并创建文件名
        dirName = usage_data
        emotionName = emotions[str(emotion_data)]
        
        # 图片要保存的文件夹
        imagePath = os.path.join(dirName,emotionName)
        
        # 创建分类文件夹以及表情文件夹
        createDir(dirName)
        createDir(imagePath)
        
        # 图片文件名
        imageName = os.path.join(imagePath,str(index)+".jpg")
        
        # 保存图片
        cv2.imwrite(imageName,image)
        imageCount = index
    print("总共有"+str(imageCount)+"张图片")

if __name__ == "__main__":
    saveImageFromFer2013("fer2013.csv")

