
# coding: utf-8

# ## 递归 读取 字典

# In[1]:

# maze = \
# {
#     9:{
#         4:{
#             9:'end'}
#     },
#     15:{
#         10:{
#             -5:'end'
#         },
#         20:{
#             -10:{
#                 5:{
#                     6:'end'},
#                 7:{
#                     -7:'end'}
#             }
#         }
#     }
# }

maze = {
    3:{
        50:{
            1:'END'
        }
    },
    4:{
        -50:{
            9:'END'
        }
    }
}


# In[2]:

# def read_maze(maze,total_reward = 0):     
#     if not isinstance(maze,dict):       
#         print("Game over. Get total reward {}".format(total_reward))
#     else:
#         action = max(maze.keys())
#         print("Take action to next state, and get immidiate reward {}".format(action))
#         read_maze(maze[action],total_reward + action)        
# read_maze(maze)      


# In[3]:

def policy(cur_state,total_reward = 0):
    if (not isinstance(cur_state,dict)):
        print("Finished the game with total reward of {}".format(total_reward))
    else:
        new_state = max(cur_state.keys())
        print("Taking action to get to state {}".format(new_state))
        policy(cur_state[new_state],total_reward+new_state)
    
policy(maze)


# In[4]:

# list1=['a']
# list2=['b']
# list1+list2

def flat_map(array):
    new_array = []
    for a in array:
        if isinstance(a,list):
            new_array += flat_map(a)
        else:
            new_array.append(a)
    return new_array


# In[5]:

def create_dict(flat_array):
    head, *tail = flat_array
    
    if(len(tail) == 1):
        return {head:tail[0]}
    else:
        return {head: create_dict(tail)}
        


# In[6]:

def invert_dict(dictionary, stack = None):
    if not stack:
        stack =[]
    if(not isinstance(dictionary, dict)):
        return dictionary
    for k,v in dictionary.items():
        stack.append([invert_dict(v),k])
    return stack


# In[7]:

def create_new_maze(dictionary):
    new_maze={}
    for path in invert_dict(dictionary):
        new_maze.update(create_dict(flat_map(path)[1:]))
    return new_maze


# In[8]:

def policy_back(cur_state):
    upside_down_maze = create_new_maze(cur_state)
#     print(upside_down_maze)
    states = []
    
    while(isinstance(upside_down_maze,dict)):
        new_state = max(upside_down_maze.keys())
        
        states = [new_state] + states
        
        upside_down_maze = upside_down_maze[new_state]
        
    states = [upside_down_maze] + states
#     print(states)
    
    total_reward = 0
    for s in states:
        total_reward += s
        print("Taking action to get to state {}".format(s))
    print("Finished the game with total reward of {}".format(total_reward))
    


# In[9]:

maze


# In[10]:

create_new_maze(maze)


# In[11]:

policy_back(maze)


# In[12]:

policy(maze)


# In[13]:

def discounted_reward(cur_state,gamma =0.9):
    if isinstance(cur_state,dict):
        return sum([k +gamma*discounted_reward(v) for k,v in cur_state.items()])
    else:
        return 0
    


# In[14]:

def policy_bellman(cur_state,total_reward =0, gamma = 0.9):
#     maze ={}
    if(not isinstance(cur_state,dict)):
        print("Finished the game with a total reward of {}".format(total_reward))
    else:
        bellman_maze ={(k+gamma*discounted_reward(v),k):v for k,v in cur_state.items()}
        
        new_state = max(bellman_maze.keys())
        
        print("Taking action to get to state {} ({})".format(new_state[1],new_state[0]))
        
        policy_bellman(bellman_maze[new_state],total_reward+new_state[1])
#         maze = bellman_maze
#     return maze


# In[16]:

print(maze)
print('-'*45)
policy(maze)
print('-'*45)
policy_back(maze)
print('-'*45)
policy_bellman(maze)

