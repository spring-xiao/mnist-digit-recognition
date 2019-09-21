# mnist-digit-recognition
## 1.DNN网络建模
网络结构
- layer1:100 nodes  
- layer2:20 nodes 
- layer3:10 nodes
- active function:Relu,Relu,softmax
- learning rate:0.0001
- regularization:L2
- iters:1000

三种优化算法的模型效果如下表，优化算法的排序为:adam>momentum>gd（梯度下降）

optimizer|train_acc|test_acc
----|:----:|:-----:
gd|0.9263|0.9293
momentum|0.9798|0.9745
adam|0.9893|0.9794




## 2.CNN网络建模
