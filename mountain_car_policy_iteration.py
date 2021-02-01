import gym
import numpy as np
import math

x_num = 50
v_num = 50

class ValueFunction():
    def __init__(self):
        self.value_m = np.zeros((x_num, v_num))
        # for i in range(x_num):
        #     self.value_m[i] = i * self.value_m[i]
        self.policy_m = np.ones((x_num, v_num, 3)) # 1*1*3represent action left, stop and right , (1,1,1) (1,1,0) (1,0,0)
        self.v_scale = 0.14/v_num
        self.x_scale = 1.8/x_num
        self.position = 0
        self.velocity = 0
        self.force = 0.001
        self.gravity = 0.0025
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.over = False
        self.e = 0.0003
        print(self.value_m)
    def update_value(self):
        over = False
        while not over:
            value_temp = self.value_m
            for i in range(x_num):
                for j in range(v_num):
                    self.position = -1.2 + i * self.x_scale
                    self.velocity = -0.07 + j * self.v_scale
                    if sum(self.policy_m[i][j]) == 3:
                        for a in range(1,4):
                            position, velocity, reward, done = self.value_step(a-1)
                            value_temp[i][j] += (reward + self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1])/3
                            if done:
                                value_temp[i][j] += 5/3
                    elif sum(self.policy_m[i][j]) == 2:
                        for a in range(1,4):
                            if self.policy_m[i][j][a-1] != 0:
                                position, velocity, reward, done = self.value_step(a-1)
                                value_temp[i][j] += (reward + self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1])/2
                                if done:
                                    value_temp[i][j] += 5/2
                    else:
                        for a in range(1,4):
                            if self.policy_m[i][j][a-1] == 1:
                                position, velocity, reward, done = self.value_step(a-1)
                                value_temp[i][j] += reward + self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1]
                                if done:
                                    value_temp[i][j] += 5
                    delta = abs(value_temp[i][j]-self.value_m[i][j])
                    if delta < self.e:
                        over = True
            self.value_m = value_temp

    def update_policy(self):
        self.over = True
        for i in range(x_num):
            for j in range(v_num):
                self.position = -1.2 + i * self.x_scale
                self.velocity = -0.07 + j * self.v_scale
                new_p = []
                m = -900000
                for a in range(0,3):
                    position, velocity, reward, done = self.value_step(a)
                    if m == self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1]:
                        new_p.append(a)
                    elif m<self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1]:
                        m = self.value_m[int((position+1.2)//self.x_scale)-1][int((velocity+0.07)//self.v_scale)-1]
                        new_p = []
                        new_p.append(a)
                num = len(new_p)
                pre_policy = self.policy_m[i][j]
                self.policy_m[i][j] = np.zeros((1, 3))
                for a in new_p:
                    self.policy_m[i][j][a] = 1/num
                for a in range(3):
                    if pre_policy[a] != self.policy_m[i][j][a]:
                        self.over = False
                        break

    def policy_iteration(self):
        while not self.over:
            self.update_value()
            self.update_policy()

    def value_step(self, action): # action =[0, 1, 2] 0left 1stop 2right
        position = self.position
        velocity = self.velocity
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position == self.min_position and velocity < 0):
            velocity = 0

        done = bool(
            position >= self.goal_position and velocity >= 0
        )
        reward = -1.0

        return position, velocity, reward, done




env = gym.make('MountainCar-v0')
val = ValueFunction()
val.policy_iteration()
print(val.policy_m)
print(val.value_m)
env.reset()
done = False
while not done:
    env.render()
    position, velocity = env.state
    action_lib = val.policy_m[int((position+1.2)//val.x_scale)-1][int((velocity+0.07)//val.v_scale)-1]
    action = 0
    for i in range(0, 3):
        if action_lib[i] == 0:
            continue
        elif action_lib[i] == 0.33333333:
            action = int(np.random(0,3))
            print(action)
        elif action_lib[i] == 0.5:
            if i == 0:
                action = 0
            elif i == 1:
                action = 2
        elif action_lib[i] == 1:
            action = i
    state, reward, done, n = env.step(action)
    position, velocity = state


