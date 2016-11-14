from __future__ import print_function
from collections import deque

from pg_reinforce import PolicyGradientREINFORCE
import tensorflow as tf
import numpy as np
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.train.SummaryWriter("/tmp/{}-experiment-10".format(env_name))

state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n
discount_factor = 0.99

def policy_network(states):
  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, 20],
                       initializer=tf.random_normal_initializer())
  b1 = tf.get_variable("b1", [20],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
  W2 = tf.get_variable("W2", [20, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("b2", [num_actions],
                       initializer=tf.constant_initializer(0))
  p = tf.matmul(h1, W2) + b2
  return p

pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       num_actions,
                                       discount_factor=discount_factor,
                                       summary_writer=writer)
NUM_ITR = 1000
BATCH_SIZE = 100
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for i_itr in xrange(NUM_ITR):
  episodes = []
  total_rewards = 0
  for i_batch in xrange(BATCH_SIZE):
    # initialize
    state = env.reset()
    rewards, states, actions, returns = [], [], [], []
    for t in xrange(MAX_STEPS):
      env.render()
      action = pg_reinforce.sampleAction(state[np.newaxis,:])
      next_state, reward, done, _ = env.step(action)
      reward = -10 if done else 0.1 # normalize reward
      ### appending the experience
      states.append(state)
      actions.append(action)
      rewards.append(reward)

      total_rewards += reward

      state = next_state
      if done: break

    return_so_far = 0
    for reward in rewards[::-1]:
      return_so_far = reward + discount_factor * return_so_far
      returns.append(return_so_far)
    #return is calculated in reverse direction
    returns = returns[::-1]

    episodes.append({
    "states" : states,
    "actions" : actions,
    "rewards" : rewards,
    "returns" : returns}
    )

  # prepare input
  states = np.concatenate([p["states"] for p in episodes])
  actions = np.concatenate([p["actions"] for p in episodes])
  returns = np.concatenate([p["returns"] for p in episodes])
  rewards = np.concatenate([p["rewards"] for p in episodes])

  inputs = [states, actions, returns]
  pg_reinforce.updateModel(inputs)

  episode_history.append(total_rewards)
  mean_rewards = np.mean(episode_history)

  print("iteration {}".format(i_itr))
  #print("Finished after {} timesteps".format(t+1))
  print("Reward for this iteration: {}".format(total_rewards))
  print("Average reward for last 100 iterations: {}".format(mean_rewards))
  if mean_rewards >= 195.0 and len(episode_history) >= 100:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break
