import numpy as np
import gym
import math
import pickle
import torch

training_episode = 2000
discount = 0.7
learning_rate = 1e-2
# theta = np.zeros(7)
#theta = np.ones(7)
#theta = np.random.random(9)
all_score = []

model = torch.nn.Sequential(
    torch.nn.Linear(8, 20),
    torch.nn.Linear(20, 3),
)


def transform(s):
    position, velocity = s
    min_position, min_velocity = env.observation_space.low
    max_position, max_velocity = env.observation_space.high
    position = (position - min_position) / (max_position - min_position) * 50
    velocity = (velocity - min_velocity) / (max_velocity - min_velocity) * 50
    return int(position), int(velocity)


def get_policy(state):
    policy = torch.zeros(3)
    for i in range(3):
        x = torch.tensor([state[0], state[1], pow(state[0], 2), pow(state[1], 2), state[0] * state[1], i,
                            i * state[0], i * state[1]]).float()
        policy[i] = model(x)
    policy = torch.softmax(policy, 0, float)
    return policy

env = gym.make('MountainCar-v0')
for i in range(training_episode):
    model.zero_grad()
    state = transform(env.reset())
    episode_state = []
    episode_action = []
    episode_reward = []
    episode_policy = []
    score = 0
    if i % 500 == 0 and i != 0:
        print('episode %d, the highest score %d' % (i, all_score[np.argmax(all_score)]))
    done = False
    while not done:
        episode_state.append(state)
        with torch.no_grad():
            policy = get_policy(state).numpy()
        print(policy)
        action_take = np.random.choice([0, 1, 2], p=policy)
        # print(action_take)
        # print(policy)
        episode_action.append(action_take)
        state, reward, done, _ = env.step(action_take)
        score += reward
        state = transform(state)
        episode_reward.append(reward)
    episode_length = len(episode_reward)
    all_score.append(score)
    for j in range(episode_length):
        t_return = 0
        for k in range(j, episode_length):
            t_return += (discount ** (k - j)) * episode_reward[k]
        j_x = get_policy(episode_state[j])[episode_action[j]]
        j_xx = torch.log(j_x)
        j_xx.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * t_return * param.grad

print('Training finished!')
with open('policy_gradient_MonteCarlo.pickle', 'wb') as f:
    pickle.dump(model, f)
    print('model saved')

# test the trained model
state = transform(env.reset())
while True:
    env.render()
    action = torch.argmax(get_policy(state))
    state, reward, done, _ = env.step(action)
    state = transform(state)
    if done:
        break