import tensorflow as tf
import numpy as np
import gym
from dqn_agent import DQNAgent
from pg_reinforce import PolicyGradientREINFORCE
from replay_buffer import ReplayBuffer

# Environment parameters
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n

# Policy nework parameters
policy_session = tf.Session()
policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
policy_writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/policy/")

def show_image(array):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(array)
    plt.title("Reward Progress")
    plt.show()

def policy_network(states):
   """ define policy neural network """
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

pg_reinforce = PolicyGradientREINFORCE(policy_session,
                                       policy_optimizer,
                                       policy_network,
                                       state_dim,
                                       summary_writer=policy_writer)

# Initializing Sampler
sampler = Sampler(pg_reinforce, env)

# Q-network parameters
q_session = tf.Session()
q_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
q_writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/q/")

def action_masker(array):
    masked_action = np.zeros((array.size, num_actions), dtype=np.float32)
    masked_action[np.arange(array.size), array] = 1.0
    return masked_action

def q_network(states):
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

dqn_agent = DQNAgent(q_session,
                     q_optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     summary_writer=q_writer)

# Initializing ReplayBuffer
buffer_size = 100000
replay_buffer = ReplayBuffer(buffer_size)
