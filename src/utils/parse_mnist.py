import numpy as np
import struct
import matplotlib.pyplot as plt

train_images_file='data/train-images.idx3-ubyte'
train_labels_file='data/train-labels.idx1-ubyte'
test_images_file='data/t10k-images.idx3-ubyte'
test_labels_file='data/t10k-labels.idx1-ubyte'

def parse_mnist_data():
##****************训练集变量数据解析
    train_images=open(train_images_file,'rb').read()
    offset=0
    fmt_header='>iiii'
    magic_num,num_images,num_row,num_col=struct.unpack_from(fmt_header, train_images, offset)
    print("训练集:"+"魔数："+str(magic_num)+"   "+"图像数量:"+str(num_images)+"   "+"图像宽："+str(num_col)+"   "+"图像长:"+str(num_col))
    fmt_image='>'+str(num_row*num_col)+'B'
    offset+=struct.calcsize(fmt_header)
    train_x=np.zeros((num_images,num_row*num_col),'int32')
    for i in range(0,num_images):
        train_x[i,]=x  =np.array(struct.unpack_from(fmt_image,train_images,offset))
        offset+=struct.calcsize(fmt_image)

##***************训练集标签数据解析
    train_labels=open(train_labels_file,'rb').read()
    offset=0
    fmt_header='>ii'
    magic_num,num_labels=struct.unpack_from(fmt_header,train_labels,offset)
    print("训练集:"+"魔数："+str(magic_num)+"   "+"标签数量:"+str(num_labels))

    fmt_label='>1B'
    offset+=struct.calcsize(fmt_header)
    train_y=np.zeros((1,num_labels),'int32')

    for i in range(0,num_labels):
        train_y[0,i]=np.squeeze(np.array(struct.unpack_from(fmt_label,train_labels,offset)))
        train_y[0,i]=int(train_y[0,i])
        offset+=struct.calcsize(fmt_label)

#****************测试集变量数据解析
    test_images=open(test_images_file,'rb').read()
    offset=0
    fmt_header='>iiii'
    magic_num,num_images,num_row,num_col=struct.unpack_from(fmt_header, test_images, offset)
    print("测试集:"+"魔数："+str(magic_num)+"   "+"图像数量:"+str(num_images)+"   "+"图像宽："+str(num_col)+"   "+"图像长:"+str(num_col))
    fmt_image='>'+str(num_row*num_col)+'B'
    offset+=struct.calcsize(fmt_header)
    test_x=np.zeros((num_images,num_row*num_col),'int32')
    for i in range(0,num_images):
        test_x[i,]=x  =np.array(struct.unpack_from(fmt_image,test_images,offset))
        offset+=struct.calcsize(fmt_image)

##***************测试集标签数据解析
    test_labels=open(test_labels_file,'rb').read()
    offset=0
    fmt_header='>ii'
    magic_num,num_labels=struct.unpack_from(fmt_header,test_labels,offset)
    print("测试集:"+"魔数: "+str(magic_num)+"   "+"标签数量:"+str(num_labels))

    fmt_label='>1B'
    offset+=struct.calcsize(fmt_header)
    test_y=np.zeros((1,num_labels),'int32')

    for i in range(0,num_labels):
        test_y[0,i]=np.squeeze(np.array(struct.unpack_from(fmt_label,test_labels,offset)))
        test_y[0,i]=int(test_y[0,i])
        offset+=struct.calcsize(fmt_label)   
    
    return train_x,train_y,test_x,test_y   