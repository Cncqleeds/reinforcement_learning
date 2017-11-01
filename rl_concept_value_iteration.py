
# coding: utf-8

# ## 通过一个简单的例子，理解增强学习的基本概念
# 
# * MDP (S,A,P,γ,R)   
# * S 
# * A
# * P
# * R
# * γ
# 
# * state value function
# * state iteration
# 

# ### 模型如下
# *  k = 1 state values as follow table
# 
# |0 | 1 | 2 | 3
# -|-|-|-
# 0 | 0 | 0 | 0 |+1
# 1|0 |# |0 |-1
# 2|0 |0 |0 |0    
# 
# 
# 
# *  k = 10 state values as follow table
# 
# |0 | 1 | 2 | 3
# -|-|-|-
# 0 | .57 | .71 | .83 |+1
# 1|.45 |# |.54 |-1
# 2|.33 |.32 |.41 |.20 
# 
# *  k = 100 state values as follow table
# 
# |0 | 1 | 2 | 3
# -|-|-|-
# 0 | .92| .94 | .96 |+1
# 1|.90 |# |.83 |-1
# 2|.88 |.86 |.84 |.71 

# In[1]:

import numpy as np
s = np.array(range(11)) #list(range(11))
a = np.array(range(5))#list(range(4))

s_dict = {0:11,1:12,2:13,3:21,4:23,5:31,6:32,7:33,8:34,9:14,10:24}
a_dict = {0:'up',1:'down',2:'left',3:'right',4:'self'}

gamma = 0.99
p = np.zeros(shape=[len(s),len(a),len(s)],dtype=np.float32)
r = np.zeros(shape=[len(s)],dtype=np.float32)
r[10] = -1.0
r[9] = 1.0
state_of_action = {11:['down','right','self'],12:['left','right','self'],13:['down','left','right','self'],14:['self'],                  21:['up','down','self'],23:['up','down','right','self'],24:['self'],                  31:['up','right','self'],32:['right','left','self'],33:['up','left','right','self'],34:['up','left','self']}

p[0,0,0] = 0.9 # p[i,k,j]表示第i个状态（如i=0表示状态11）采取第k个动作(如k=0表示动作'up')后得到第j个状态（j=0表示状态11）概率为1
p[0,0,1] = 0.1

p[0,1,0] = 0.1
p[0,1,1] = 0.1
p[0,1,3] = 0.8

p[0,2,0] = 0.9
p[0,2,3] = 0.1

p[0,3,0] = 0.1
p[0,3,1] = 0.8
p[0,3,3] = 0.1

p[0,4,0] = 1.0
##  for state 11

p[1,0,1] = 0.8
p[1,0,0] = 0.1
p[1,0,2] = 0.1

p[1,1,1] = 0.8
p[1,1,0] = 0.1
p[1,1,2] = 0.1


p[1,2,0] = 0.8
p[1,2,1] = 0.2

p[1,3,1] = 0.2
p[1,3,2] = 0.8

p[1,4,1] = 1.0
##  for state 12 

p[2,0,2] = 0.8
p[2,0,1] = 0.1
p[2,0,9] = 0.1

p[2,1,1] = 0.1
p[2,1,9] = 0.1
p[2,1,4] = 0.8

p[2,2,1] = 0.8
p[2,2,2] = 0.1
p[2,2,4] = 0.1

p[2,3,2] = 0.1
p[2,3,4] = 0.1
p[2,3,9] = 0.8

p[2,4,2] = 1.0

p[3,0,0] = 0.8
p[3,0,3] = 0.2

p[3,1,3] = 0.2
p[3,1,5] = 0.8

p[3,2,0] = 0.1
p[3,2,3] = 0.8
p[3,2,5] = 0.1

p[3,3,3] = 0.8
p[3,3,0] = 0.1
p[3,3,5] = 0.1

p[3,4,3] = 1.0

p[4,0,2] = 0.8
p[4,0,4] = 0.1
p[4,0,10] = 0.1

p[4,1,4] = 0.1
p[4,1,7] = 0.8
p[4,1,10] = 0.1

p[4,2,2] = 0.1
p[4,2,4] = 0.8
p[4,2,7] = 0.1

p[4,3,2] = 0.1
p[4,3,7] = 0.1
p[4,3,10] = 0.8

p[4,4,4] = 1.0

p[5,0,3] = 0.8
p[5,0,5] = 0.1
p[5,0,6] = 0.1

p[5,1,5] = 0.9
p[5,1,6] = 0.1

p[5,2,5] = 0.9
p[5,2,3] = 0.1

p[5,3,3] = 0.1
p[5,3,5] = 0.1
p[5,3,6] = 0.8

p[5,4,5] = 1.0

p[6,0,5] = 0.1
p[6,0,6] = 0.8
p[6,0,7] = 0.1

p[6,1,5] = 0.1
p[6,1,6] = 0.8
p[6,1,7] = 0.1

p[6,2,5] = 0.8
p[6,2,6] = 0.2

p[6,3,6] = 0.2
p[6,3,7] = 0.8

p[6,4,6] = 1.0

p[7,0,6] = 0.1
p[7,0,4] = 0.8
p[7,0,8] = 0.1

p[7,1,6] = 0.1
p[7,1,7] = 0.8
p[7,1,8] = 0.1

p[7,2,6] = 0.8
p[7,2,4] = 0.1
p[7,2,7] = 0.1

p[7,3,4] = 0.1
p[7,3,8] = 0.8
p[7,3,7] = 0.1

p[7,4,7] = 1.0

p[8,0,10] = 0.8
p[8,0,7] = 0.1
p[8,0,8] = 0.1

p[8,1,7] = 0.1
p[8,1,8] = 0.9

p[8,2,7] = 0.8
p[8,2,8] = 0.1
p[8,2,10] = 0.1

p[8,3,8] = 0.9
p[8,3,10] = 0.1

p[8,4,8] = 1.0

p[9,0,9] = 1.0

p[9,1,9] = 1.0

p[9,2,9] = 1.0

p[9,3,9] = 1.0

p[9,4,9] = 1.0

p[10,0,10] = 1.0

p[10,1,10] = 1.0

p[10,2,10] = 1.0

p[10,3,10] = 1.0

p[10,4,10] = 1.0

## ....

steps = 1000

def value_iteration(v_cur,s,a,p,gamma,r):
    v = np.zeros(shape=[len(s)],dtype=np.float32)
    for i in range(len(s)):
        reward_a = np.zeros(shape=[len(a)],dtype = np.float32)
        for k in range(len(a)):
            sum_future_reward = 0
            for j in range(len(s)):
                sum_future_reward = sum_future_reward + p[i,k,j]*v_cur[j]
            reward_a[k] = sum_future_reward
        temp = np.max(reward_a)
            
        v[i] = r[i] + gamma * temp
    return v

        
def mdp(s,a,p,gamma,r):
    epsilon = 1e-4
    step = 0
    v_cur = np.zeros(shape=[len(s)],dtype=np.float32)
    v_new = value_iteration(v_cur,s,a,p,gamma,r)
    step = step + 1
    dist = np.linalg.norm(v_cur-v_new)
    
    while (dist>epsilon) and (step<steps):
        v_cur = v_new
        v_new = value_iteration(v_cur,s,a,p,gamma,r)
        step = step + 1
        dist = np.linalg.norm(v_cur-v_new)
        
    v_new = v_new/(max(v_new) + 1e-8) # 归一化
    return v_new
    
print("State set:\n",s,"\n")
print("Action set:\n",a,"\n")
print("Transition probability matrix shape:\n",p.shape,"\n")
print("Immediate Reward of state:\n",r,"\n")
print("Optimal state value function  v* after %d steps iteration:\n"%(steps),mdp(s,a,p,gamma,r))

