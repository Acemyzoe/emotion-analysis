# emotion-analysis  表情分析

**BY** [GJ](https://github.com/Acemyzoe/emotion-analysis.git)

## 使用指南 
1. 运行emotion_analysis_model.py用于构建、训练、保存网络，并可以对单张人脸的照片进行分析。 
2. 运行emotion-analysis-from-video.py进行表情实时分析。
3. Face_extraction.py作为提取照片人脸和处理图片的模块，将在emotion_analysis_model.py中使用，将图片中的人脸提取后再分析表情可以提升很多的准确率。同时也可以单独运行，截取某张图片中的人脸。
4. data文件夹保存了一些示例图片，fer2013文件中保存了网络训练数据集fer2013.csv，可以运行csv2jpg.py将矩阵格式转换成图片形式。
5. model文件夹保存了网络模型。json文件保存模型的结构，h5文件保存模型的权重参数。xml文件是提取人脸的分类器。LOG文件中记录了一些模型训练的准确率以供参考。

## 环境配置
  * ubuntu18.04+python3.7
  * 推荐使用Anaconda（一个提供包管理和环境管理的python版本）。  [官网下载](https://www.anaconda.com/distribution/)
  * 推荐修改镜像地址：
  
      >pip install pip -U  
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  
* 安装需要的python库：(缺少相应的库可用conda或者pip自行安装) 
    > * opencv  
    > * matplotlib
    >  * keras
