# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import pygame
import random

from constant import Color as C

REWARD_EVENT = pygame.USEREVENT #创建即时奖励事件

def quitGame():
#    pygame.time.set_timer(ACTION_EVENT,0)
    pygame.quit()
    quit()


class Block(pygame.sprite.Sprite): #继承用于碰撞检测的Sprite类
    width = 32 # 方块宽度
    height = 32 # 方块高度
    def __init__(self,color,name,bg_size,head_height):
        pygame.sprite.Sprite.__init__(self)
        self.bg_size = bg_size
        self.head_height = head_height
        self.bg_width,self.bg_height = self.bg_size
        self.color = color
        self.name = name
        self.surface = pygame.Surface((Block.width,Block.height))
        self.surface.fill(color)
        self.rect = self.surface.get_rect()
        self.rect.left = random.randint(0,self.bg_width - self.rect.width)
        self.rect.top = random.randint(0,self.bg_height - self.rect.height - self.head_height)
        
class Food(Block):
    init_energy = 1000 # 初始能量
    range_speed = 9 # 速度范围
    blood_height = 2 # 血槽的高度
    def __init__(self,color,name,bg_size,head_height):
        Block.__init__(self,color,name,bg_size,head_height)
        self.energy = Food.init_energy
        self.speed = [random.randint(-Food.range_speed,Food.range_speed),\
                      random.randint(-Food.range_speed,Food.range_speed)]
        self.movable = False #是否可移动 或正在移动中
        self.energy_remain = 1.0 #剩余能量比例
        
    def move(self):
        if self.movable:
            if self.rect.left<0 or self.rect.right > self.bg_width:
                self.speed[0] = -self.speed[0]
            elif self.rect.top <0 or self.rect.bottom > self.bg_height-self.head_height:
                self.speed[1] = -self.speed[1]
    
            self.rect.move_ip(self.speed)
            self.energy -=1
        
        self.energy_remain = self.energy/Food.init_energy
    
class Animal(Block):
    range_speed = 15
    def __init__(self,color,name,bg_size,head_height):
        Block.__init__(self,color,name,bg_size,head_height)
        self.rect.left = self.bg_width//2
        self.rect.top = self.bg_height // 3
        self.speed = [random.choice([-Animal.range_speed,Animal.range_speed]),\
                      random.choice([-Animal.range_speed,Animal.range_speed])]
        self.movable = False
        self.active = True #生命状态
        
    def move(self):
        if self.active and self.movable:
            if (self.rect.left<0) or (self.rect.right > self.bg_width) or self.rect.top <0 or self.rect.bottom > self.bg_height-self.head_height:
                self.active = False   
                
            self.rect.move_ip(self.speed)

class Game():
    def __init__(self,bg_size = (84*8,84*8),frame_rate = 60,delay_max = 100):
        pygame.init()
        self.bg_size = bg_size
        self.bg_width,self.bg_height = self.bg_size
        self.head_height = 60
        self.frame_rate = frame_rate
        self.actions = 5
        self.rendering = False # 显示状态
        self.screen = pygame.Surface(self.bg_size)
        self.screen_rect = self.screen.get_rect()
        self.screen.fill(C.BLACK)
        self.clock = pygame.time.Clock()
        self.delay = 0
        self.delay_max = delay_max
        self.food_num = 3
        
        
        self.score = 0 #总得分
        self.reward = 0 # 即时得分 
        self.im_reward = 0 # 用于强化机器学习 
        self.info_font = pygame.font.Font(None,30) 
        
        # 设置信息栏
        self.head_surf = pygame.Surface((self.bg_width,self.head_height))
        self.head_rect = self.head_surf.get_rect()
        
        self.score_surf = self.info_font.render("Score:" + str(self.score),True,C.BLACK,C.WHITE)
        self.score_rect = self.score_surf.get_rect()
        
        self.reward_surf = self.info_font.render("Reward:"+ str(self.reward),True,C.BLACK,C.WHITE)
        self.reward_rect = self.reward_surf.get_rect() 
        
        # 设置游戏界面
        self.body_surf = pygame.Surface((self.bg_width, self.bg_height - self.head_rect.height))
        self.body_rect = self.body_surf.get_rect()
        self.done = False
        self.running = True
        
    def _ceate_screen(self):
        ######### 开始绘制到screen上
        self.screen.fill(C.BLACK)
        self.body_surf.fill(C.BLACK)
        for each in self.surfs:
            self.body_surf.blit(each.surface,each.rect)
            
        #为每个食物绘制血槽
        for each in self.foods:
            pygame.draw.line(self.body_surf,C.WHITE,\
                         (each.rect.left,each.rect.top - Food.blood_height -2),\
                         (each.rect.right-1,each.rect.top -Food.blood_height-2),2)
            energy_color = C.WHITE
            if each.energy_remain > 0.2:
                energy_color = C.GREEN
            else:
                energy_color = C.RED
                
            pygame.draw.line(self.body_surf,energy_color,\
                             (each.rect.left,each.rect.top - Food.blood_height-2),\
                             (each.rect.left +each.rect.width* each.energy_remain-1,each.rect.top -Food.blood_height-2),2)
                
        #把信息栏和游戏主界面绘画到screen 
        self.screen.blit(self.head_surf,self.head_rect)
        self.screen.blit(self.body_surf,self.body_rect)
        
    def reset(self):
        self.done = False
        self.rnning = True
        self.score = 0
        self.reward = 0
        
        self.score_surf = self.info_font.render("Score:" + str(self.score),True,C.BLACK,C.WHITE)
        self.score_rect = self.score_surf.get_rect()
        
        self.reward_surf = self.info_font.render("Reward:"+ str(self.reward),True,C.BLACK,C.WHITE)
        self.reward_rect = self.reward_surf.get_rect() 
        
        ## 很明确 就是 得到 self.screen, 然后返回
        self.animal = Animal(C.RED,"Animal",self.bg_size,self.head_height)
        self.animal.active = True
        self.animal.movable = False
        self.surfs = []
        self.foods =[]
        
        self.group = pygame.sprite.Group() 
        self.surfs.append(self.animal) # 添加动物到surfs
        
        #生成食物
        self.group.add(self.animal)
        for i in range(self.food_num):
            food = Food(C.YELLOW,"Food",self.bg_size,self.head_height)
            while pygame.sprite.spritecollide(food,self.group,False):
#                print("collided!!")
                food = Food(C.YELLOW,"Food",self.bg_size,self.head_height)
            food.movable = False
            self.foods.append(food)
            self.surfs.append(food)
        self.group.add(food)
        self.group.remove(self.animal)
    
        self.score_rect.left = 10
        self.score_rect.top =10
          
        self.reward_rect.right = self.bg_width - 10
        self.reward_rect.top = 10
    
        self.head_surf.fill(C.BLACK)
        #在信息栏下方划条横线
        pygame.draw.line(self.head_surf,C.WHITE,(0,self.head_height-4),(self.bg_width,self.head_height-4),4)
        # 将score 和reward 贴在信息栏中
        self.head_surf.blit(self.score_surf,self.score_rect)
        self.head_surf.blit(self.reward_surf,self.reward_rect)
     
        self.body_rect.top =self.head_rect.height   
        self.body_surf.fill(C.BLACK)
        
        ######### 开始绘制到screen上
        self._ceate_screen()
        #返回当前Screen的像素RGB值，用于强化学习
        
        train_screen = pygame.transform.smoothscale(self.screen,(84,84))
#        pygame.image.save(save_screen, "../images/"+"screen"+str(time.time())+".png")
        return pygame.surfarray.array3d(train_screen)
    
    def step(self,action):
        ## 修改screen 然后返回 s_, reward, done
        
        if action == 0: # UP
            event = pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_UP})
            pygame.event.post(event)
        elif action == 1: # DOWN
            event = pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_DOWN})
            pygame.event.post(event)
        elif action == 2:
            event = pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_LEFT})
            pygame.event.post(event)
        elif action == 3: 
            event = pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_RIGHT})
            pygame.event.post(event)
        elif action == 4: # 无操作 或叫做 空闲操作
            self.animal.move()
            
            for each in self.foods:
                each.move()
                # 食物之间的碰撞检测处理
                self.group.remove(each)
                if pygame.sprite.spritecollide(each,self.group,False):
                    each.speed[0] = - each.speed[0]
                    each.speed[1] = - each.speed[1]
                self.group.add(each)
                #将没有能力的食物从列表中删除
                if each.energy_remain<0:
                    self.foods.remove(each)
                    self.surfs.remove(each)
                    self.group.remove(each)
                    
            if not self.foods:
                self.done = True
                self.running = False
            if not self.animal.active:
                self.done = True
                self.running = False
                
            # 动物开始吃食物
            for each in  pygame.sprite.spritecollide(self.animal,self.group,False):
                self.group.remove(each)
                self.surfs.remove(each)
                self.foods.remove(each)
                self.score += each.energy
                self.reward = each.energy
                
                #创建并发起一个时间 用来获取即时奖励
                event = pygame.event.Event(REWARD_EVENT,{"im_reward":self.reward})
                pygame.event.post(event)
                print("这时应该获得到奖励")
                            
                self.score_surf = self.info_font.render("Score:" + str(self.score),True,C.BLACK,C.WHITE)
                self.score_rect = self.score_surf.get_rect()
                self.score_rect.left = 10
                self.score_rect.top =10
                
                self.reward_surf = self.info_font.render("Reward:"+ str(self.reward),True,C.BLACK,C.WHITE)
                self.reward_rect = self.reward_surf.get_rect()
                self.reward_rect.right = self.bg_width - 10
                self.reward_rect.top = 10
                
                self.head_surf = pygame.Surface((self.bg_width,self.head_height))
                self.head_rect = self.head_surf.get_rect()
                
                self.head_surf.fill(C.BLACK)
                pygame.draw.line(self.head_surf,C.WHITE,(0,self.head_height-4),(self.bg_width,self.head_height-4),4)
                self.head_surf.blit(self.score_surf,self.score_rect)
                self.head_surf.blit(self.reward_surf,self.reward_rect)
                
                #生成新的食物            
                food = Food(C.YELLOW,"Food",self.bg_size,self.head_height)
                food.movable = self.animal.movable
                
                self.group.add(self.animal)
                while pygame.sprite.spritecollide(food,self.group,False):
#                    print("collided!!")
                    food = Food(C.YELLOW,"Food",self.bg_size,self.head_height)
                    food.movable = self.animal.movable   
                self.group.remove(self.animal)
                    
                food.move()
                self.foods.append(food)
                self.surfs.append(food)
                self.group.add(food)
                    
        elif action == 5: # SPACE
            event = pygame.event.Event(pygame.KEYDOWN,{'key':pygame.K_SPACE})
            pygame.event.post(event)
          
            
#        global imm_reward 
        imm_reward = 0
        # 如果发生即时奖励事件，获得即时奖励
        for event in pygame.event.get(REWARD_EVENT):
            imm_reward = event.im_reward
            print("奖励来了")    
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitGame()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.animal.movable = not self.animal.movable
                    for f in self.foods:
                        f.movable = not f.movable
                elif event.key == pygame.K_UP:
                    self.animal.speed = [0, -Animal.range_speed]
                elif event.key == pygame.K_DOWN:
                    self.animal.speed = [0,Animal.range_speed]
                elif event.key == pygame.K_LEFT:
                    self.animal.speed = [-Animal.range_speed,0]
                elif event.key ==pygame.K_RIGHT:
                    self.animal.speed = [Animal.range_speed,0]
     
        
        
        self._ceate_screen()
        #返回当前Screen的像素RGB值，用于强化学习      
        train_screen = pygame.transform.smoothscale(self.screen,(84,84))
        return pygame.surfarray.array3d(train_screen), imm_reward/Food.init_energy, self.done
    
    def render(self):
        # 仅做一件事，把 screen  贴到 sceen_mode上
        if not self.rendering:
            self.rendering = True
            self.screen_mode = pygame.display.set_mode(self.bg_size)
            pygame.display.set_caption("JustEat   -- Cncqleeds")
            self.screen_mode.blit(self.screen,self.screen_rect)
            pygame.display.update()
            
        else:            
            self.screen_mode.blit(self.screen,self.screen_rect)
            pygame.display.update()

                
#g = Game()
#s = g.reset()
#print(s.shape)
#
#while g.animal.active:
##    g.step(random.)
#    s_,imm_r,done = g.step(random.randint(0,6))
#    
#    g.render()
#    print(done)
#    if imm_r:
#        print(imm_r)
#    print(g.animal.rect.top)
#    g.clock.tick(60)
#
#print("Game over!")
            

