import numpy as np
import gym
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#调用pytorch的分布函数
FixedCategorical = torch.distributions.Categorical
#重新定义原来的分布类的成员函数
old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)
log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs=lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).\
    sum(-1).unsqueeze(-1)
FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)


class Sample:
    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net
        self.gamma = 0.9
    def sample_episodes(self, num_episodes):
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        for i in range(num_episodes):
            observation = self.env.reset()
            episode_rewards = []
            while True:
                state = torch.from_numpy(np.reshape(observation, [1, 2])).float()
                action = self.policy_net.decide_action(state)
                action = action.numpy()[0, 0]
                observation_, reward, done, _ = self.env.step(action)
                batch_obs.append(observation)
                batch_actions.append(action)
                episode_rewards.append(reward)
                if done:
                    reward_sum = 0
                    discount_reward_sum = np.zeros_like(episode_rewards)
                    for j in reversed(range(0, len(episode_rewards))):
                        reward_sum = reward_sum*self.gamma + episode_rewards[j]
                        discount_reward_sum[j] = reward_sum
                    discount_reward_sum -= np.mean(discount_reward_sum)
                    discount_reward_sum /= np.std(discount_reward_sum)
                    for t in range(len(episode_rewards)):
                        batch_rewards.append(discount_reward_sum[t])
                    break
                observation = observation_
        batch_obs = np.reshape(batch_obs, [len(batch_obs), self.policy_net.feature_nums])
        batch_actions = np.reshape(batch_actions, [len(batch_actions), 1])
        batch_rewards = np.reshape(batch_rewards, [len(batch_rewards), 1])
        batch_obs = torch.from_numpy(batch_obs).float()
        batch_actions = torch.from_numpy(batch_actions).float()
        batch_rewards = torch.from_numpy(batch_rewards).float()
        return batch_obs, batch_actions, batch_rewards


class Linear_Layer(nn.Module):
    def __init__(self, input_num, layer_size=20):
        super(Linear_Layer, self).__init__()
        self.layer_size = layer_size
        self.linear1 = nn.Linear(input_num, self.layer_size)
        nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear1.bias, 0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        hidden_layer = F.relu(x)
        return hidden_layer

    @property
    def output_size(self):
        return self.layer_size


class Categorical_Layer(nn.Module):
    def __init__(self, input_num, output_num):
        super(Categorical_Layer, self).__init__()
        self.linear2 = nn.Linear(input_num, output_num)
        nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear2.bias, 0.1)

    def forward(self, inputs):
        x = self.linear2(inputs)
        return  FixedCategorical(logits=x)


class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.learning_rate = 0.1
        self.feature_nums = env.observation_space.shape[0]
        print(self.feature_nums)
        self.action_num = env.action_space.n
        self.base = Linear_Layer(self.feature_nums)
        self.dist = Categorical_Layer(self.base.output_size, self.action_num)

    def decide_action(self, inputs, deterministic=False):
        linear_layer_output = self.base(inputs)
        categorical_layer_output = self.dist(linear_layer_output)
        if deterministic:
            action = categorical_layer_output.mode()
        else:
            action = categorical_layer_output.sample()
        return action

    def action_log(self, inputs, action):
        linear_layer_output = self.base(inputs)
        categorical_layer_output = self.dist(linear_layer_output)
        action_log_probs = categorical_layer_output.log_probs(action)
        return action_log_probs


class Policy_Gradient:
    def __init__(self, policy_net, lr = 0.01):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def update(self, obs_batch, action_batch, reward_batch):
        action_log_probs = -self.policy_net.action_log(obs_batch, action_batch)
        loss = (action_log_probs*reward_batch).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def policy_train(env, alg, training_num):
    reward_sum = 0
    reward_sum_line = []
    training_time = []
    for i in range(training_num):
        sampler = Sample(env, alg.policy_net)
        temp = 0
        training_time.append(i)
        train_obs, train_actions, train_rewards = sampler.sample_episodes(1)
        alg.update(train_obs, train_actions, train_rewards)
        if i == 0:
            reward_sum = policy_test(env, alg.policy_net, False, 1)
        else:
            reward_sum = 0.9*reward_sum + 0.1*policy_test(env, alg.policy_net, False, 1)
        reward_sum_line.append((reward_sum))
        print(reward_sum)
        print("training episodes is %d,trained reward_sum is %f" % (i, reward_sum))
        if reward_sum > -150:
            break
    plt.plot(training_time, reward_sum_line)
    plt.xlabel("training number")
    plt.ylabel("score")
    plt.show()


def policy_test(env, policy, render, test_num):
    for i in range(test_num):
        observation = env.reset()
        reward_sum = 0
        while True:
            if render:
                env.render()
            state = np.reshape(observation, [1, 2])
            state= torch.from_numpy(state).float()
            action = policy.decide_action(state, deterministic=True)
            action = action.numpy()[0, 0]
            observation_, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
            observation = observation_
    return reward_sum


if __name__=='__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.unwrapped
    env.seed(1)
    policy_net = Policy(env)
    policy_gradient = Policy_Gradient(policy_net, lr=0.1)
    training_num = 15000
    policy_train(env, alg=policy_gradient, training_num=training_num)
    reward_sum = policy_test(env, policy_net, True, 10)