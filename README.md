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

采用学习率阶梯指数递减，adam优化算法，迭代1000次,模型效果并未提升，模型效果：
- train_acc: 0.98945
- test_acc: 0.9794

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

![markdown](https://github.com/spring-xiao/mnist-digit-recognition/blob/master/result/acc-model-dnn-img.jpg)

以测试集的accuracy为评价指标，选出的最优模型效果如下，模型文件保存为model/mnist-model-dnn.pkl
- train_acc: 0.9884
- test_acc: 0.9806

最优模型的混淆矩阵

训练集混淆矩阵

|   |0|1|2|3|4|5|6|7|8|9|acc_rate
----|----|----|----|----|----|----|----|----|----|----|----
|0|5876|2|5|1|1|3|15|3|12|5|0.992065
|1|1|6694|17|4|8|0|3|9|5|1|0.99288
|2|13|10|5883|6|8|0|3|20|12|3|0.987412
|3|2|3|33|6022|0|21|1|16|20|13|0.982221
|4|2|11|5|0|5784|0|13|2|4|21|0.990072
|5|5|3|1|15|2|5354|21|0|8|12|0.987641
|6|10|9|0|0|7|8|5874|0|10|0|0.992565
|7|2|24|18|2|12|0|2|6183|4|18|0.986911
|8|8|20|7|11|4|8|11|2|5773|7|0.986669
|9|8|3|1|11|30|10|1|17|9|5859|0.984871

测试集混淆矩阵

|   |0|1|2|3|4|5|6|7|8|9|acc_rate
----|----|----|----|----|----|----|----|----|----|----|----
|0|973|0|1|0|1|0|2|1|2|0|0.992857
|1|0|1123|4|1|0|1|3|1|2|0|0.989427
|2|4|1|1012|2|2|0|3|6|2|0|0.98062
|3|0|0|3|992|0|4|0|6|4|1|0.982178
|4|3|0|1|0|960|0|4|1|1|12|0.977597
|5|3|1|0|7|1|872|4|1|2|1|0.977578
|6|4|3|0|0|4|3|941|0|3|0|0.982255
|7|1|9|10|2|1|0|0|997|0|8|0.969844
|8|4|0|1|3|4|1|4|2|954|1|0.979466
|9|4|3|0|5|9|1|0|4|1|982|0.973241



## 2.CNN网络建模

网络结构
- layer1: (conv,filter:5$\times$5$\times$32,stride:1,padding:same,active funtion:Relu)
- layer2: (pool,filter:3$\times\$3,stride:1,padding:valid,pool type:max)
- layer3: (conv,filter:5$\times$5$\times$64,stride:1,padding:same,active function:Relu)
- layer4: (pool,filter:5$\times\$5,stride:1,padding:valid,pool type:max)
- layer5: (full connection,nodes:100,active function:relu)
- layer6: (full connection,nodes:10,active function:softmax)

$\sqrt{x^{2}}$ 
5$\times$5$


