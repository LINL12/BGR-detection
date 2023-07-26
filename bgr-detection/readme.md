# ResNet18+图像二分类+pytorch 

博客地址：[ResNet18+图像二分类+pytorch_LINL631的博客-CSDN博客](https://blog.csdn.net/L_IN_L/article/details/131939035?csdn_share_tail={"type"%3A"blog"%2C"rType"%3A"article"%2C"rId"%3A"131939035"%2C"source"%3A"L_IN_L"})



## 项目背景

最近在检查项目的时候发现有部分的图像通过opencv打开保存后自动保存为了BGR图像，而windoms系统打开查看是默认RGB的，所以会造成一定的影响

![image-20230726111739515](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726111739515.png)

可以在上图看到这种现象，所以我训练了一个简单是二分类模型来对BGR和RGB图像进行分类，使用的是pytorch框架，考虑到轻量化，所以Resnet18模型



## 项目大纲

![image-20230726112159717](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726112159717.png)



## 数据处理

对于这种简单的二分类模型，数据集的处理相对较为简单。

我采用的是类似coco数据集的方式来对路径进行管理：（BGR-detection/bgr-detection/data/bgr-data.yaml）

![](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726112849795.png)



训练集（BGR-detection/dataset/image/train）里存放BGR和RGB图片：

![image-20230726114103761](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726114103761.png)



我是使用txt文件来存放他们的标签的：

![image-20230726113301064](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726113301064.png)



当然，如果你也对图片进行和相应类别的命名的话也可以参考如下的方式来获取标签：

![image-20230726113542174](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726113542174.png)

这种方式通过读取图片的名称信息来赋标签



> > > 下面的各部分代码通过模块化的方式来编写，便于后期的管理和调整<<<<



## 数据读取/加载

数据加载模块：（BGR-detection/bgr-detection/utils/dataLoader.py)

![image-20230726114447263](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726114447263.png)



## 定义模型

线性层的输出神经元个数对应要分类的类别数量：(BGR-detection/bgr-detection/model/ResNet18.py)

![image-20230726114642456](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726114642456.png)



## 模型训练

定义训练过程：（BGR-detection/bgr-detection/utils/trainresnet.py)

![image-20230726115724148](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726115724148.png)

![image-20230726114923153](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726114923153.png)

![image-20230726115601130](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726115601130.png)



主程序训练：(/data0/linhao/BGR-detection/bgr-detection/train.py)

![image-20230726115418718](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726115418718.png)



## 检测

因为我的需求是对一整个文件夹中的图像进行分类，并将结果分类存放，所以没有设置过多的应用场景，大家可以根据自身需修改：(BGR-detection/bgr-detection/detection.py)

![image-20230726115905771](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726115905771.png)

![image-20230726115920466](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726115920466.png)

## 运行示例

运行前：

请确保代你的路径设置正确，

请确保你们数据和标签相对应（避免浪费时间训练一个无用的模型），

请确保各模块代码被放置在正确位置并且被正确的调用



运行示例如下：

![image-20230726142840093](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726142840093.png)



我是在服务器上运行的，所以是Linux命令，在编译器上运行同理



每一次训练结果和测试结果都会被默认保存：

![image-20230726143114508](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726143114508.png)



![image-20230726143204860](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726143204860.png)





我的训练集不大，就800张图片，包含了两个类别，机器是3090的显卡，训练了35个epochs，用时一个小时左右，准确率可以保证在95以上

torch版本信息：

![image-20230726144019292](C:\Users\16095\AppData\Roaming\Typora\typora-user-images\image-20230726144019292.png)









## Author

因为是不常见任务，所以数据集我就不放上来了（估计你们也不需要），需要的话私信我



Design by LINL



