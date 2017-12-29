# -*- coding: utf-8 -*-
import justeat1

class Env():
    def __init__(self,render_flag=False):
        self.render_flag = render_flag 
        self.game = justeat1.Game()
        self.actions = self.game.actions
        self.reset()
             
    def reset(self):
        ## 返回初始化的界面 screen 
        s = self.game.reset()
        if self.render_flag:
            self.render()
        return s
        
    def render(self):
        ## 显示当前的 screen
        self.game.render()
        
    def step(self, action):
        ## 返回 s_, reward, done
        s_,reward,done = self.game.step(action)
        if self.render_flag:
            self.render()
        return s_,reward,done
        