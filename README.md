# mnist-digit-recognition
## 1.DNN网络建模
(1) 网络结构和基本超参数配置
- layer1: 100 nodes  
- layer2: 20 nodes 
- layer3: 10 nodes
- active function: Relu,Relu,softmax
- learning rate: 0.0001
- regularization: L2
- iters: 1000

三种优化算法的模型效果如下表，优化算法的排序为:adam>momentum>gd（梯度下降）

optimizer|train_acc|test_acc
----|:----:|:-----:
gd|0.9263|0.9293
momentum|0.9798|0.9745
adam|0.9893|0.9794

采用学习率阶梯指数递减，adam优化算法，迭代1000次,模型效果 train_acc:0.98945,test_acc:0.9794,模型效果并未提升。

(2) 网络结构和基本超参数配置
- layer1: 100 nodes  
- layer2: 50 nodes 
- layer3: 10 nodes
- active function: Relu,Relu,softmax
- learning rate: 0.0001,
- decay_rate: 0.9
- optimizer: adam
- regularization:L2
- iters:1000

学习率阶梯指数递减，以50次为一阶梯，记录每次训练的模型效果，训练集和测试集的效果图：

![markdown](https://github.com/spring-xiao/mnist-digit-recognition/tree/master/result/acc-model-dnn-img.jpg)

## 2.CNN网络建模
