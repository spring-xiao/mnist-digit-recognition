import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import pickle

class DNN:
 
    iters = 10000
    tol = 1e-3
    batch_size = 64
    task_type = 'binary'
    learning_rate = 0.0001
    learning_decay = False
    learning_decay_rate = 0.9
    regularizer_type = 'l2'
    regularizer_rate = 0.5
    optimizer = 'gd'
    momentum = 0.9
    beta1 = 0.9
    beta2 = 0.999
    layer_dims = [4,4,1]
    keep_probs = [1.0,1.0,1.0]
    active_functions = ['relu','relu','sigmoid']
    parameters = None
    cost = {'iter':[],'all_cost':[],'entropy_cost':[]}
    seed = 100
    
    def __init__(self,iters = 10000,tol = 1e-3,task_type = 'binary',
                 batch_size = 64,
                 learning_rate = 0.5,
                 learning_decay = False,
                 learning_decay_rate = 0.9,
                 regularizer_type = 'l2',
                 regularizer_rate = 0.0001,
                 optimizer = 'gd',
                 momentum = 0.9,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 layer_dims = [4,4,1],
                 keep_probs = [1.0,1.0,1.0],
                 active_functions = ['relu','relu','sigmoid'],
                 parameters = None):
        
        assert isinstance(layer_dims,list),'参数layer_dims需要为list类型'
        assert isinstance(keep_probs,list),'参数keep_probs需要为list类型'
        assert isinstance(active_functions,list),'参数active_functions需要为list类型'
        assert len(layer_dims) == len(active_functions),'参数:layer_dims和active_functions长度不一致'
        assert len(layer_dims) == len(keep_probs),'参数:layer_dims和keep_probs长度不一致'
        assert task_type in['binary','multi'],'task_type，请选择binary和multi之一'
        self.iters = iters
        self.tol = tol
        self.batch_size = batch_size
        self.task_type = task_type
        self.learning_rate = float(learning_rate)
        self.learning_decay = learning_decay
        self.learning_decay_rate = float(learning_decay_rate)
        self.regularizer_type = regularizer_type
        self.regularizer_rate = float(regularizer_rate)
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.layer_dims = layer_dims
        self.active_functions = active_functions
        self.keep_probs = keep_probs
        self.keep_probs[-1] = 1.0
        if self.task_type == 'binary':
            self.layer_dims[-1] = 1
            self.active_functions[-1] = 'sigmoid'
            
        if self.task_type == 'multi':
            self.active_functions[-1] = 'softmax'
        
        self.parameters = parameters
            
    def __str__(self):
        
        attr  =  []
        for item in self.__dict__.items():
            if list(item)[0] not in ['parameters','cost']:
                attr.append(item)
       
        attr  =  ','.join(['%s = %s' %item for item in attr])
        attr  =  'DNN('+attr+')'
        
        return(attr)
    
    __repr__  =  __str__
        
    def get_params(self):
        print(self.iters)
        print(self.tol)
    
    
    @classmethod
    def one_hot(cls,Y,depth,axis = 0):
        
        tf.reset_default_graph()
        one_hot = tf.one_hot(Y,depth,axis = axis)
        with tf.Session() as sess:
            Y_label = sess.run(one_hot)
        Y_label = np.squeeze(Y_label)
        return(Y_label)
    
    
    
    
    def shuffle_batch(self,data,seed = 10):
        
        train_X = data['X']
        train_Y = data['Y']
        
        n_sample = train_X.shape[1]
        np.random.seed(seed)
        
        shuffle = np.random.permutation(n_sample)
        train_X = train_X[:,shuffle]
        train_Y = train_Y[:,shuffle]
        
        n_batch = n_sample//self.batch_size
        batch_data = list()
        
        for i in range(0,n_batch):
            batch_X = train_X[:,i*self.batch_size:(i+1)*self.batch_size]
            batch_Y = train_Y[:,i*self.batch_size:(i+1)*self.batch_size]
            batch_data.append((batch_X,batch_Y))
        
        if n_sample%self.batch_size  != 0:
            batch_X = train_X[:,n_batch*self.batch_size:n_sample]
            batch_Y = train_Y[:,n_batch*self.batch_size:n_sample]
            batch_data.append((batch_X,batch_Y))
    
        return(batch_data)
        
        
    def create_placehold(self,data):
        
        X = data['X']
        Y = data['Y']
        
        n_x = X.shape[0]
        n_y = Y.shape[0]
        
        X = tf.placeholder(tf.float32,shape = (n_x,None),name = 'X')
        Y = tf.placeholder(tf.float32,shape = (n_y,None),name = 'Y')
        
        return(X,Y)
        
    
    def create_constant_parameters_tensor(self,parameters):
    
        n_layer = len(parameters)//2

        parameters_tensor = {}
        for i in range(1,n_layer+1):
            w_temp = tf.constant(parameters['W'+str(i)],tf.float32,shape = parameters['W'+str(i)].shape)
            b_temp = tf.constant(parameters['b'+str(i)],tf.float32,shape = parameters['b'+str(i)].shape)

            parameters_tensor['W'+str(i)] = w_temp
            parameters_tensor['b'+str(i)] = b_temp

        return parameters_tensor 
    
    
    def init_parameters(self,X,seed = None):
        
        if (isinstance(X,tf.Tensor)):
            n_x = X.shape[0].value
        else:
            n_x = X.shape[0]
            
        n_layer = list([n_x])+self.layer_dims
        parameters = dict()
        for i in range(1,len(n_layer)):
            np.random.seed(seed)
            value_w = np.random.randn(n_layer[i],n_layer[i-1])*np.sqrt(2.0/(n_layer[i]+n_layer[i-1]))
            value_b = np.zeros([n_layer[i],1])
            
            param_w = tf.Variable(initial_value = value_w,trainable = True,name = 'W'+str(i),dtype = tf.float32)
            param_b = tf.Variable(initial_value = value_b,trainable = True,name = 'b'+str(i),dtype = tf.float32)
            
            parameters['W'+str(i)] = param_w
            parameters['b'+str(i)] = param_b
       
        return parameters
    
    def assign_parameters(self):
        
        if self.parameters is None:
            assert 1 == 2,'parameters为None，请为其指定值'
        n_layers = len(self.layer_dims)
        parameters = dict()
        for i in range(1,n_layers+1):
            
            value_w = self.parameters['W'+str(i)]
            value_b = self.parameters['b'+str(i)]
            
            param_w = tf.Variable(initial_value = value_w,trainable = True,name = 'W'+str(i),dtype = tf.float32)
            param_b = tf.Variable(initial_value = value_b,trainable = True,name = 'b'+str(i),dtype = tf.float32)
            
            parameters['W'+str(i)] = param_w
            parameters['b'+str(i)] = param_b
            
        return parameters
         
    def forward_propagation(self,parameters,X):
        
        n_layer = len(parameters)//2
        
        A = {'A0':X}
        
        for i in range(1,n_layer):
            z_temp = tf.add(tf.matmul(parameters['W'+str(i)],A['A'+str(i-1)]),
                          parameters['b'+str(i)])
        
            if self.active_functions[i] == 'relu':
                A_temp = tf.nn.relu(z_temp)
            elif self.active_functions[i] == 'tanh':
                A_temp = tf.nn.tanh(z_temp)
            elif self.active_functions[i] == 'sigmoid':
                A_temp = tf.nn.sigmoid(z_temp)
            elif self.active_functions[i] == 'softplus':
                A_temp = tf.nn.softplus(z_temp)
            else:
                A_temp = tf.nn.relu(z_temp)
            
            if self.regularizer_type == 'dropout':
                A_temp = tf.nn.dropout(A_temp,self.keep_probs[i-1])
            
            A['A'+str(i)] = A_temp
        
        logits = tf.add(tf.matmul(parameters['W'+str(n_layer)],A['A'+str(n_layer-1)]),
                      parameters['b'+str(n_layer)])
        
        return logits
    
    def compute_cost(self,labels,logits):
        
        labels = tf.transpose(labels)
        logits = tf.transpose(logits)
 
        if self.task_type == 'binary':
            cost_matrix = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels,logits = logits)
        elif self.task_type == 'multi':
            cost_matrix = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels,logits = logits)
        
        cost = tf.reduce_mean(cost_matrix)
        
        return cost
        
    def compute_cost_with_regularizer(self,labels,logits,parameters):
        
        n_layer = len(parameters)//2
    
        cost_entropy = self.compute_cost(labels = labels,logits = logits)
        if self.regularizer_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(self.regularizer_rate)
        elif self.regularizer_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)
        
        cost_regularizer = tf.constant(0,dtype = tf.float32)
        for i in range(1,n_layer+1):
            cost_regularizer = tf.add(regularizer(parameters['W'+str(i)]),cost_regularizer)
        
        cost = tf.add(cost_entropy,cost_regularizer)
        
        return cost,cost_entropy
        
  
    def fit(self,data,param_type = 'assign',init_param_seed = None,batch_seed = None,decay_step = None,is_print = True):
        
        tf.reset_default_graph()
        
        data['X'] = data['X'].astype(np.float32)
        data['Y'] = data['Y'].astype(np.float32)
        
        (X,Y) = self.create_placehold(data)
        
        if self.parameters is None:
            parameters = self.init_parameters(X = X,seed = init_param_seed)
        elif param_type == 'assign':
            parameters = self.assign_parameters()
        else:
            parameters = self.init_parameters(X = X,seed = init_param_seed)
            
        
        logits = self.forward_propagation(parameters = parameters,X = X)
        
        global_step = tf.Variable(0,dtype = tf.float32,trainable = False)
        
        if self.learning_decay:
            
            n_sample = tf.shape(data['X'])[1]
            n_sample = tf.to_float(n_sample)
            if decay_step is None:
                decay_steps = tf.math.ceil(tf.div(n_sample,self.batch_size))            
            else:
                decay_steps = decay_step*tf.math.ceil(n_sample/self.batch_size)
            
            learning_rate = tf.train.exponential_decay(learning_rate = self.learning_rate,
                                                     global_step = global_step,
                                                     decay_steps = decay_steps,
                                                     decay_rate = self.learning_decay_rate,
                                                     staircase = True)
        else:
            learning_rate = tf.constant(self.learning_rate,dtype = tf.float32)
            
        if self.regularizer_type in ['l1','l2']:
            cost,cost_entropy = self.compute_cost_with_regularizer(labels = Y,logits = logits,parameters = parameters)
        
        else:
            cost = self.compute_cost(labels = Y,logits = logits)
            cost_entropy = cost
        
        if self.optimizer == 'gd':
            Optimizer_type = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost,global_step = global_step)
            
        elif self.optimizer == 'momentum':
            Optimizer_type = tf.train.MomentumOptimizer(learning_rate = learning_rate,
                                                      momentum = self.momentum).minimize(cost,global_step = global_step)
        
        elif self.optimizer == 'adam':
            
            Optimizer_type = tf.train.AdamOptimizer(learning_rate = learning_rate,
                                                   beta1 = self.beta1,
                                                   beta2 = self.beta2).minimize(cost,global_step = global_step)
        
        else:
            Optimizer_type = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost,global_step = global_step)
        
        init_var = tf.initializers.global_variables()
        
        with tf.Session() as sess:
            sess.run(init_var)
     
            self.cost['iter'] = []
            self.cost['all_cost'] = []
            self.cost['entropy_cost'] = []
            for i in range(0,self.iters):
                
                batch_data = self.shuffle_batch(data,seed = batch_seed)
                n_sample = data['X'].shape[1]
                #iter_cost = 0.0
                finished_sample = 0
                
                if is_print:
                    sys.stdout.write('epoches:{0}/{1}'.format(i+1,self.iters))
                    sys.stdout.write('\n')
                
                for minibatch in batch_data:
                    
                    (batch_X,batch_Y) = minibatch
                    _,batch_cost_all,batch_cost_entropy,y_prob = sess.run([Optimizer_type,cost,cost_entropy,logits],feed_dict = {X:batch_X,Y:batch_Y})
                    #iter_cost+= batch_cost/len(batch_data)
                    
                    cost_ = round(batch_cost_entropy*1,6)
                    finished_sample+= batch_X.shape[1]
                    pct = round(finished_sample*100/n_sample,1)
                    
                            
                    if self.task_type == 'binary':
                        y_prob = 1.0/(1+np.exp(-y_prob))
                        y_labels_pre = y_prob>= 0.5
                        y_labels_pre = y_labels.astype('int32')
                        y_labels_pre = np.squeeze(y_labels_pre)
                        y_labels_true = np.squeeze(batch_Y)
                        
                        
                    elif self.task_type == 'multi':
                        
                        y_prob = np.exp(y_prob)/(np.sum(np.exp(y_prob),axis = 0))
                        y_labels_pre = np.argmax(y_prob,axis = 0)   
                        y_labels_true = np.argmax(batch_Y,axis = 0) 
                        
                        
                    else:
                        assert 1 == 2,'task_type(binary or multi)任务不明确'    
                     
                    acc = round(np.sum(y_labels_pre == y_labels_true)*100/len(y_labels_true),3)
                    
                    if is_print:
                        output_str = f'{finished_sample}/{n_sample} [finished:{pct}%] - loss:{cost_} - accuracy:{acc}%    \r'
                        sys.stdout.flush()
                        sys.stdout.write(output_str)
                
                if is_print:
                    sys.stdout.write('\n')
                    
                if (i+1)%10 == 0 or (i+1) == 1:
                    self.cost['iter'].append(i+1)
                    #self.cost['cost'].append(round(iter_cost,6))
                    self.cost['all_cost'].append(round(batch_cost_all,6))
                    self.cost['entropy_cost'].append(round(batch_cost_entropy,6))
                  
                self.parameters = sess.run(parameters) 


    def __forward_propagation(self,parameters,X):
        
        n_layer = len(parameters)//2
        
        A = {'A0':X}
        
        for i in range(1,n_layer):
            z_temp = tf.add(tf.matmul(parameters['W'+str(i)],A['A'+str(i-1)]),
                          parameters['b'+str(i)])
        
            if self.active_functions[i] == 'relu':
                A_temp = tf.nn.relu(z_temp)
            elif self.active_functions[i] == 'tanh':
                A_temp = tf.nn.tanh(z_temp)
            elif self.active_functions[i] == 'sigmoid':
                A_temp = tf.nn.sigmoid(z_temp)
            elif self.active_functions[i] == 'softplus':
                A_temp = tf.nn.softplus(z_temp)
            else:
                A_temp = tf.nn.relu(z_temp)
            
            A['A'+str(i)] = A_temp
        
        logits = tf.add(tf.matmul(parameters['W'+str(n_layer)],A['A'+str(n_layer-1)]),
                      parameters['b'+str(n_layer)])
        
        return logits                
                
    def predict_prob(self,X,batch_size = 32):
        
        assert self.parameters != None,'未拟合参数'
        assert X.shape[0] == self.parameters['W1'].shape[1],'输入集X和权重参数W1不匹配'
        
        (n_vars,n_sample) = X.shape
        cnt = int(n_sample/batch_size)
        
        tf.reset_default_graph()
        
        # X_tensor = tf.constant(X,dtype = tf.float32,shape = X.shape,name = 'X')
        
        X_tensor = tf.placeholder(dtype = tf.float32,shape = (n_vars,None),name = 'X')
        parameters = self.create_constant_parameters_tensor(self.parameters)
                      
        logits = self.__forward_propagation(parameters = parameters,X = X_tensor)
        logits = tf.transpose(logits)
        
        if self.task_type == 'binary':
            Y_prob = tf.nn.sigmoid(logits)
        
        elif self.task_type == 'multi':
            Y_prob = tf.nn.softmax(logits)
            
        else:
            assert 1 == 2,'task_type(binary or multi)任务不明确'
        
        with tf.Session() as sess:
            
            for i in range(0,cnt):
                
                batch_X = X[:,i*batch_size:(i+1)*batch_size]
                batch_prob = sess.run(Y_prob,feed_dict = {X_tensor:batch_X})
                if i == 0:
                    Y_prob_pre = batch_prob
                else:
                    Y_prob_pre = np.concatenate((Y_prob_pre,batch_prob),axis = 0)
                
                
                finished_sample = (i+1)*batch_size
                finished_pct = round((i+1)*batch_size*100/n_sample,2)
                output_str = f'{finished_sample}/{n_sample} [finished:{finished_pct}%]     \r'
                sys.stdout.flush()
                sys.stdout.write(output_str)
                
            if cnt*batch_size<n_sample:
                
                batch_X = X[:,cnt*batch_size:n_sample]
                batch_prob = sess.run(Y_prob,feed_dict = {X_tensor:batch_X})
                if cnt == 0:
                    Y_prob_pre = batch_prob
                else:
                    Y_prob_pre = np.concatenate((Y_prob_pre,batch_prob),axis = 0)
                
                output_str = f'{n_sample}/{n_sample} [finished:100.00%]    \r'
                sys.stdout.flush()
                sys.stdout.write(output_str)
            
            sys.stdout.write('\n')     
        
        Y_prob_pre = Y_prob_pre.T
        
        return Y_prob_pre
    
    def predict(self,X,batch_size = 32):
        
        assert self.parameters != None,'未拟合参数'
        assert X.shape[0] == self.parameters['W1'].shape[1],'输入集X和权重参数W1不匹配'
        
        tf.reset_default_graph()
        
        (n_vars,n_sample) = X.shape
        cnt = int(n_sample/batch_size)
        
        # X_tensor = tf.constant(X,dtype = tf.float32,shape = X.shape,name = 'X')
        X_tensor = tf.placeholder(dtype = tf.float32,shape = (n_vars,None),name = 'X')
        parameters = self.create_constant_parameters_tensor(self.parameters)
                      
        logits = self.__forward_propagation(parameters = parameters,X = X_tensor)
        logits = tf.transpose(logits)
        
        if self.task_type == 'binary':
            Y_prob = tf.nn.sigmoid(logits)
            Y_labels = tf.math.less_equal(tf.constant(0.5,dtype = tf.float32),Y_prob)
            Y_labels = tf.to_int32(Y_labels)
            Y_labels = tf.squeeze(Y_labels)
        
        elif self.task_type == 'multi':
            Y_prob = tf.nn.softmax(logits)
            Y_labels = tf.argmax(Y_prob,axis = 1,output_type = tf.int32)
            
        else:
            assert 1 == 2,'task_type(binary or multi)任务不明确'
        
        with tf.Session() as sess:
            
            #Y_labels = sess.run(Y_labels)
            for i in range(cnt):
                
                batch_X = X[:,i*batch_size:(i+1)*batch_size]
                batch_labels = sess.run(Y_labels,feed_dict = {X_tensor:batch_X})
                if i == 0:
                    Y_labels_pre = batch_labels
                else:
                    Y_labels_pre = np.concatenate((Y_labels_pre,batch_labels),axis = 0)
                
                finished_sample = (i+1)*batch_size
                finished_pct = round((i+1)*batch_size*100/n_sample,2)
                output_str = f'{finished_sample}/{n_sample} [finished:{finished_pct}%]     \r'
                sys.stdout.flush()
                sys.stdout.write(output_str)
            
            if cnt*batch_size < n_sample:
                
                batch_X = X[:,cnt*batch_size:n_sample]
                batch_labels = sess.run(Y_labels,feed_dict = {X_tensor:batch_X})
                if cnt == 0:
                    Y_labels_pre = batch_labels
                else:
                    Y_labels_pre = np.concatenate((Y_labels_pre,batch_labels),axis = 0)
                
                output_str = f'{n_sample}/{n_sample} [finished:100.00%]    \r'
                sys.stdout.flush()
                sys.stdout.write(output_str)
            
            sys.stdout.write('\n')     
            
        Y_labels_pre = Y_labels_pre.T
        
        return Y_labels_pre
        
    def get_log_loss(self,labels,predictions):
        
        assert self.parameters != None,'未拟合参数'
        assert labels.shape == predictions.shape,'labels和predictions维度应一致'
        tf.reset_default_graph()
        
        log_loss = tf.losses.log_loss(labels = labels,predictions = predictions)
        parameters = self.create_constant_parameters_tensor(parameters = self.parameters)
        n_len = len(parameters)//2
        
        if self.regularizer_type == 'l1':
            regularizer = tf.contrib.layers.l1_regularizer(self.regularizer_rate)
        elif self.regularizer_type == 'l2':
            regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)
        else:
            regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)
            
        regularizer_loss = tf.constant(0,dtype = tf.float32)
        for i in range(1,n_len+1):
            regularizer_loss = tf.add(regularizer(parameters['W'+str(i)]),regularizer_loss)
            
        all_loss = tf.add(log_loss,regularizer_loss)
        
        with tf.Session() as sess:
            all_loss,log_loss,regularizer_loss = sess.run([all_loss,log_loss,regularizer_loss])
    
        if self.regularizer_type == 'l1':
            loss_dict = {'all_loss':round(all_loss,6),'log_loss':round(log_loss,6),'l1_loss':round(regularizer_loss,6)}
        elif self.regularizer_type == 'l2':
            loss_dict = {'all_loss':round(all_loss,6),'log_loss':round(log_loss,6),'l2_loss':round(regularizer_loss,6)}
        else:
            loss_dict = {'all_loss':round(all_loss,6),'log_loss':round(log_loss,6),'l2_loss':round(regularizer_loss,6)}
        
        return loss_dict
        
    def set_params(self,**kwargs):
        
        attr = self.__dict__.keys()
        for var in kwargs:
            if var not in attr:
                print(var+':无效属性')
            else:
                setattr(self,var,kwargs[var])
                
    @staticmethod
    def save_model(model,filename):

        with open(filename,'wb') as file:
            pickle.dump(model,file)

        print('保存成功')

    @staticmethod
    def load_model(filename):

        with open(filename,'rb') as file:
            model = pickle.load(file)

        return model



        
                