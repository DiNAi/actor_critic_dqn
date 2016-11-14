from neural_q_learner_for_fixed_policy import NeuralQLearner
from pg_reinforce import PolicyGradientREINFORCE
from sampler import Sampler

import tensorflow as tf
import numpy as np
import gym


#Environment parameters
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n

#tensorflow sessions
q_sess = tf.Session()
policy_sess = tf.Session()

#Q-network parameters
q_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
q_writer = tf.train.SummaryWriter("/tmp/{}-experiment-1-q".format(env_name))

#policy network parameters
policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9)
policy_writer = tf.train.SummaryWriter("/tmp/{}-experiment-1-policy".format(env_name))
discount_factor = 0.99

#sampler parameters
BATCH_SIZE = 100
MAX_STEPS    = 200

def q_network(states):
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

q_learner = NeuralQLearner(q_sess,
                           q_optimizer,
                           q_network,
                           state_dim,
                           num_actions,
                           summary_writer=q_writer)

pg_reinforce = PolicyGradientREINFORCE(policy_sess,
                                       policy_optimizer,
                                       policy_network,
                                       state_dim,
                                       num_actions,
                                       discount_factor=discount_factor,
                                       summary_writer=policy_writer)

sampler = Sampler(pg_reinforce,
                  env,
                  BATCH_SIZE,
                  MAX_STEPS)

NUM_ITR = 1000

iteration_history = deque(maxlen=100)

for i_itr in xrange(NUM_ITR):
  episodes = []
  total_rewards = 0
  for i_batch in xrange(BATCH_SIZE):
    # initialize
    state = env.reset()
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for t in xrange(MAX_STEPS):
      action = pg_reinforce.sampleAction(state[np.newaxis,:])
      next_state, reward, done, _ = env.step(action)
      reward = -10 if done else 0.1 # normalize reward
      ### appending the experience
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)
      dones.append(done)

      total_rewards += reward

      state = next_state
      if done: break

    episodes.append({
    "states" : states,
    "actions" : actions,
    "rewards" : rewards,
    "next_states" : next_states,
    "dones" : dones}
    )

  # prepare input
  states = np.concatenate([p["states"] for p in episodes])
  actions = np.concatenate([p["actions"] for p in episodes])
  rewards = np.concatenate([p["rewards"] for p in episodes])
  next_states = np.concatenate([p["next_states"] for p in episodes])
  dones = np.concatenate([p["dones"] for p in episodes])
  next_action_probs = pg_reinforce.compute_action_probabilities(next_states)

  inputs = [states, actions, rewards, next_states, next_action_probs, dones]

  q_learner.storeExperience(*inputs)
  q_learner.updateModel()
  # computing returns from q_network after updating it
  returns = q_learner.q_values(states, actions)
  # update policy gradient model
  pg_reinforce.updateModel([states, actions, returns])
  # printing the progress
  iteration_history.append(total_rewards/BATCH_SIZE)
  mean_rewards = np.mean(iteration_history)#
  print("iteration {}".format(i_itr))
  print("Reward for this iteration: {}".format(total_rewards/BATCH_SIZE))
  print("Average reward for last 100 iterations: {}".format(mean_rewards))
  if mean_rewards >= 195.0 and len(episode_history) >= 100:
    print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
    break
