from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers


class DQN(object):
    def __init__(self):
        self.step = 0
        self.update_freq = 200  # 模型更新频率
        self.replay_size = 2000  # 训练集大小
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        STATE_DIM, ACTION_DIM = 2, 3
        model = models.Sequential([
            layers.Dense(100, input_dim=STATE_DIM, activation='relu'),
            layers.Dense(ACTION_DIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.Adam(0.001))
        return model

    def act(self, s, epsilon=0.1):
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.model.predict(np.array([s]))[0])

    def save_model(self, file_path='MountainCar-v0-dqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_queue.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, factor=0.95):
        if len(self.replay_queue) < self.replay_size:
            return
        self.step += 1
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        replay_batch = random.sample(self.replay_queue, batch_size)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.model.predict(s_batch)
        Q_next = self.target_model.predict(next_s_batch)

        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + factor * np.amax(Q_next[i]))

        self.model.fit(s_batch, Q, verbose=0)


env = gym.make('MountainCar-v0')
episodes = 1000
score_list = []
agent = DQN()
for i in range(episodes):
    s = env.reset()
    score = 0
    while True:
        a = agent.act(s)
        next_s, reward, done, _ = env.step(a)
        agent.remember(s, a, next_s, reward)
        agent.train()
        score += reward
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', i, 'score:', score, 'max:', max(score_list))
            break

    if np.mean(score_list[-10:]) > -160:
        agent.save_model()
        break
env.close()