
* kaggle数据集中，数据由48x48像素的面部灰度图像组成。面部已自动注册，因此面部或多或少居中，并且在每个图像中占据大约相同的空间量。任务是根据面部表情中显示的情感将每张脸分为以下七个类别之一（0 =愤怒，1 =厌恶，2 =恐惧，3 =快乐，4 =悲伤，5 =惊喜，6 =中性）。第一列为表情类别（0=angry，1=disgust，2=fear,3=happy,4=sad,5=surprise,7=nutural)，第二列为图像像素点，第三列为数据分类（Training、PublicTest、PrivateTest)


* train.csv包含两列，“情感”和“像素”。“情感”列包含图像中存在的情感的数字代码，范围从0到6（含）。“像素”列包含每个图像中用引号引起来的字符串。该字符串的内容以行主顺序以空格分隔的像素值。test.csv仅包含“像素”列，您的任务是预测情感列。


* [比赛官网 ](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
