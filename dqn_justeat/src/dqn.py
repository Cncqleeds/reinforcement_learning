import game_env
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os 
import random

env = game_env.Env()

# DQN 中的神经网络架构
class Qnetwork():
    def __init__(self,h_size):
        self.scalarInput = tf.placeholder(shape=[None,21168],\
                                         dtype = tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape = [-1,84,84,3])
        
        self.conv1 = tf.contrib.layers.convolution2d(\
                inputs = self.imageIn,\
                num_outputs =32,\
                kernel_size =[8,8],\
                stride = [4,4],\
                padding = 'VALID',\
                biases_initializer = None)
        
        self.conv2 = tf.contrib.layers.convolution2d(\
                inputs = self.conv1,\
                num_outputs =64,\
                kernel_size =[4,4],\
                stride = [2,2],\
                padding = 'VALID',\
                biases_initializer = None)
        
        self.conv3 = tf.contrib.layers.convolution2d(\
                inputs = self.conv2,\
                num_outputs =64,\
                kernel_size =[3,3],\
                stride = [1,1],\
                padding = 'VALID',\
                biases_initializer = None)
        
        self.conv4 = tf.contrib.layers.convolution2d(\
                inputs = self.conv3,\
                num_outputs =512,\
                kernel_size =[7,7],\
                stride = [1,1],\
                padding = 'VALID',\
                biases_initializer = None)
        
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        
        self.AW = tf.Variable(tf.random_normal([h_size//2,env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(\
                                    self.Advantage,reduction_indices=1,keep_dims = True))
        self.predict = tf.argmax(self.Qout,1)
        
        self.targetQ = tf.placeholder(shape=[None],dtype = tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype= tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout,self.actions_onehot),\
                               reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        

## DQN 中的memory
class experience_buffer():
    def __init__(self,buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
        
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])
        
# 状态预处理
def processState(states):
    
    return np.reshape(states,[21168])
    
#  Target网路图的参数更新
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx + total_vars//2].assign((var.value()*tau)\
                        +((1- tau)*tfVars[idx + total_vars//2].value())))
    return op_holder
# 在sess中更新Target
def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)
        
# DQN 中的超参数

batch_size = 256
update_freq = 32
y = 0.99
startE = 1
endE = 0.01
anneling_steps = 300000.0
num_episodes = 300000
pre_train_steps = 100000
max_epLength = 5000
load_model = False
path = "./dqn/"
h_size = 512
tau = 0.001

# 搭建DQN
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)
init = tf.global_variables_initializer()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

# 创建memory
myBuffer = experience_buffer()
# 定义epsilon 用于探索
e = startE
# 定义epsilon阶梯
stepDrop = (startE - endE) / anneling_steps

rList = []
total_steps = 0

# 定义 模型保存路径
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)
    
## 正式运行 Sess

with tf.Session() as sess:
    if load_model == True:
        print("Loading Model...")
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        
    sess.run(init)
    
    updateTarget(targetOps,sess)
    for i in range(num_episodes + 1):
        print("Episode %d / %d."%(i+1,num_episodes))
        epsiodeBuffer = experience_buffer()
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        while j < max_epLength:
            j += 1
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,env.actions+1)
            else:
                a = sess.run(mainQN.predict,feed_dict=\
                            {mainQN.scalarInput:[s]})[0]
#            print("action:",a)
            s_,r,d = env.step(a)
            s_ = processState(s_)
            total_steps += 1
            epsiodeBuffer.add(np.reshape(np.array([s,a,r,s_,d]),[1,5]))

            if total_steps > pre_train_steps:
                if e>endE:
                    e -= stepDrop
                if total_steps %(update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)
                    A = sess.run(mainQN.predict,feed_dict =\
                                {mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q = sess.run(targetQN.Qout,feed_dict=\
                                {targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    doubleQ = Q[range(batch_size),A]
                    targetQ = trainBatch[:,2] + y*doubleQ
                    _ = sess.run(mainQN.updateModel,feed_dict=\
                                {mainQN.scalarInput:np.vstack(trainBatch[:,0]),\
                                mainQN.targetQ:targetQ,mainQN.actions:trainBatch[:,1]})
                    updateTarget(targetOps,sess)
            rAll += r
            s = s_

            if d == True:
                break
                
        myBuffer.add(epsiodeBuffer.buffer)
        rList.append(rAll)

        if i > 0 and i%250 == 0:
            print("episode",i,", average reward of last 250 episode",\
                  np.mean(rList[-25:]))

        if i > 0 and i % 5000 == 0:
            saver.save(sess,path + '/model-'+str(i)+'.cptk')
            print("Saved Model")
            
    saver.save(sess,path + '/model-'+str(i)+'.cptk')
    
## 绘制运行结果

rMat = np.resize(np.array(rList),[len(rList)//3000,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
        
        