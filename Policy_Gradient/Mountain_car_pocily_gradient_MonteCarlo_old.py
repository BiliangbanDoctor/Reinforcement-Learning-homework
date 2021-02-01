# import numpy as np
# import gym
# import math
# import pickle
#
#
# def transform(s):
#     position, velocity = s
#     min_position, min_velocity = env.observation_space.low
#     max_position, max_velocity = env.observation_space.high
#     position = (position - min_position) / (max_position - min_position) * 50
#     velocity = (velocity - min_velocity) / (max_velocity - min_velocity) * 50
#     return int(position), int(velocity)
#
#
# def get_policy(state, theta):
#     sum = 0
#     policy = np.zeros(3)
#     for i in range(3):
#         x = np.array([1, state[0], state[1], state[0] * state[1] / 100,
#                       i - 1, (i - 1) * state[0], (i - 1) * state[1]])
#         policy[i] = math.e ** (np.matmul(x, theta))
#         sum += policy[i]
#     policy = policy / sum
#     return policy
#
#
# def get_gradient(state, theta, action):
#     x_state_a = np.array(
#         [1, state[0], state[1], state[0] * state[1] / 100,
#          action - 1, (action - 1) * state[0], (action - 1) * state[1]])
#     policy_state_sum = np.zeros(7)
#     policy = get_policy(state, theta)
#     for i in range(3):
#         policy_state_sum += policy[i] * np.array(
#             [1, state[0], state[1], state[0] * state[1] / 100,
#              (i - 1), (i - 1) * state[0], (i - 1) * state[1]])
#     #return x_state_a - policy_state_sum  # too small
#     r = x_state_a - policy_state_sum
#     return r
#
#
# def get_state_value(state, w):
#     x = np.array([1, state[0], state[1], state[0] * state[1] / 100])
#     return np.matmul(x, w)
#
#
# training_episode = 10000
# learning_rate_theta = 1e-3
# discount = 0.7
# # theta = np.zeros(7)
# theta = np.ones(7)
# w = np.ones(4)
# #theta = np.random.random(9)
# all_score = []
#
# env = gym.make('MountainCar-v0')
# for i in range(training_episode):
#     # print(theta)
#     state = transform(env.reset())
#     episode_state = []
#     episode_action = []
#     episode_reward = []
#     score = 0
#     if i % 500 == 0 and i != 0:
#         print('episode %d, the highest score %d' % (i, all_score[np.argmax(all_score)]))
#         print(w, theta)
#     done = False
#     while not done:
#         episode_state.append(state)
#
#         policy = get_policy(state, theta)
#         action_take = np.random.choice([0, 1, 2], p=policy)
#         # print(action_take)
#         # print(policy)
#         episode_action.append(action_take)
#         state, reward, done, _ = env.step(action_take)
#         score += reward
#         state = transform(state)
#         episode_reward.append(reward)
#     episode_length = len(episode_reward)
#     all_score.append(score)
#     for j in range(episode_length):
#         t_return = 0
#         for k in range(j, episode_length):
#             # t_return += (discount ** (k - j)) * episode_reward[k]
#             t_return += episode_reward[k]
#         #t_return -= get_state_value(episode_state[j], w)  # return from step t minus baseline(state_value)
#         j_state_feature = np.array([1, episode_state[j][0], episode_state[j][1],
#                                     episode_state[j][0] * episode_state[j][1] / 100])
#         #learning_rate_w = 0.1 / np.matmul(j_state_feature, j_state_feature)
#         #learning_rate_w = 1e-3
#         #w += learning_rate_w * (discount ** j) * t_return * j_state_feature
#         #print(get_gradient(episode_state[j], theta, episode_action[j]))
#         #learning_rate_theta = 1 / np.matmul(get_gradient(episode_state[j], theta, episode_action[j]), np.ones(7))
#         theta += learning_rate_theta * (discount ** j) * t_return * get_gradient(episode_state[j], theta,
#                                                                                  episode_action[j])
#
# print('Training finished!')
# with open('policy_gradient_MonteCarlo.pickle', 'wb') as f:
#     pickle.dump(theta, f)
#     print('model saved')
#
# # test the trained model
# state = transform(env.reset())
# while True:
#     env.render()
#     action = np.argmax(get_policy(state, theta))
#     state, reward, done, _ = env.step(action)
#     state = transform(state)
#     if done:
#         break

# failed pytorch version
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
