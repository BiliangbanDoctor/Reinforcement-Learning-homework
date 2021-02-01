import gym
import numpy as np
import pickle

env = gym.make('MountainCar-v0')


def transform(state):
    position, velocity = state
    min_position, min_velocity = env.observation_space.low
    max_position, max_velocity = env.observation_space.high
    position = (position - min_position) / (max_position - min_position) * 50
    velocity = (velocity - min_velocity) / (max_velocity - min_velocity) * 50
    return int(position), int(velocity)


training_episode = 10000
learning_rate = 0.7
discount = 0.9
Q = np.zeros((50, 50, 3))
epsilon = 0.2
all_score = []
for i in range(training_episode):
    state = transform(env.reset())
    # if i != 0 :
    #     print('episodes:%d, the highest score:%d' % (i, all_score[np.argmax(all_score)]))
    if i % 500 == 0 and i != 0:
        print('episodes:%d, the highest score:%d' % (i, all_score[np.argmax(all_score)]))
    score = 0
    while True:
        if np.random.random(1) < epsilon:  # epsilon-greedy policy
            a = np.random.choice([0, 1, 2])
        else:
            a = np.argmax(Q[state[0]][state[1]])
        # if i < 2000:
        #     a = np.random.choice([0, 1, 2])
        next_state, reward, done, _ = env.step(a)
        score += reward
        next_state = transform(next_state)
        next_a = np.argmax(Q[next_state[0]][next_state[1]])
        Q[state[0]][state[1]][a] = (1 - learning_rate) * Q[state[0]][state[1]][a] + learning_rate * (
                    reward + discount * Q[next_state[0]][next_state[1]][next_a])
        if done:
            all_score.append(score)
            break
        state = next_state
with open('action_state_value.pickle', 'wb') as f:
    pickle.dump(Q, f)
    print('model saved')

# test the trained policy
for i in range(10):
    state = transform(env.reset())
    while True:
        env.render()
        a = np.argmax(Q[state[0]][state[1]])
        state, reward, done, _ = env.step(a)
        state = transform(state)
        if done:
            break
