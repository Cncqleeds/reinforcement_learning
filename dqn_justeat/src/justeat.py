# -*- coding: utf-8 -*-
import pygame


from constant import Color as C

class Game():
    def __init__(self,bg_size = (84*8,84*8),frame_rate = 60,delay_max = 100):
        pygame.init()
        self.bg_size = bg_size
        self.frame_rate = frame_rate
        self.actions = 5
        self.rendering = False # 显示状态
        self.screen = pygame.Surface(self.bg_size)
        self.screen_rect = self.screen.get_rect()
        self.screen.fill(C.BLACK)
        self.clock = pygame.time.Clock()
        self.delay = 0
        self.delay_max = delay_max
        
    def reset(self):
        ## 很明确 就是 得到 self.screen, 然后返回
        
        pass
    
    def step(self,action):
        ## 修改screen 然后返回 s_, reward, done
        
        self.delay = (self.delay+1) % self.delay_max

        if self.delay % 50 == 0:
            self.screen.fill(C.RED)
        elif self.delay % 60 == 0:
            self.screen.fill(C.GREEN)
    
    def render(self):
        # 仅做一件事，把 screen  贴到 sceen_mode上
        if not self.rendering:
            self.rendering = True
            self.screen_mode = pygame.display.set_mode(self.bg_size)
            self.screen_mode.blit(self.screen,self.screen_rect)
            pygame.display.update()
            
        else:            
            self.screen_mode.blit(self.screen,self.screen_rect)
            pygame.display.update()

                
g = Game()
#g.render()
while True:
    g.step(0)
    g.render()
    print(g.delay)
    g.clock.tick(60)

print("nothing")
            
