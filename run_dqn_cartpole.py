from __future__ import print_function
from collections import deque

from neural_q_learner_for_fixed_policy import NeuralQLearner
import tensorflow as tf
import numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.train.SummaryWriter("/tmp/{}-experiment-1".format(env_name))

state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n

def target_policy(states):
    #define target policy
    c1 = tf.constant([[1.,2.], [1.,1.5], [0.,2.], [1.,2.]])
    c2 = tf.matmul(states, c1)
    # print('states', states)
    # print('print states', states)
    # print('c1', sess.run(c1))
    # print('sum_prob', sess.run(tf.reduce_sum(c1, reduction_indices=1, keep_dims=True)))

    action_prob = c2/tf.reduce_sum(c2, reduction_indices=1, keep_dims=True)
    print(action_prob)
    return action_prob


def observation_to_action(states):
  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("W2", [20, num_actions],
                       initializer=tf.random_normal_initializer())
  b2 = tf.get_variable("b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  q = tf.matmul(h1, W2) + b2
  return q

q_learner = NeuralQLearner(sess,
                           optimizer,
                           observation_to_action,
                           target_policy,
                           state_dim,
                           num_actions,
                           summary_writer=writer)

MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for i_episode in xrange(MAX_EPISODES):

  # initialize
  state = env.reset()
  total_rewards = 0

  for t in xrange(MAX_STEPS):
    env.render()
    action = env.action_space.sample()    #### behavior policy that uniformally choose an action for {0, 1}.
    next_state, reward, done, _ = env.step(action)

    total_rewards += reward
    # reward = -10 if done else 0.1 # normalize reward
    q_learner.storeExperience(state, action, reward, next_state, done)

    q_learner.updateModel()
    state = next_state

    if done: break

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("Episode {}".format(i_episode))
  print("Finished after {} timesteps".format(t+1))
  print("Reward for this episode: {}".format(total_rewards))
  print("Average reward for last 100 episodes: {}".format(mean_rewards))
  if mean_rewards >= 195.0:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break
