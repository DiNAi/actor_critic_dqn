import tensorflow as tf
from dqn_agent import DQNAgent
import gym

env = gym.make("CartPole-v0")

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

session = tf.Session()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

dqn_agent = DQNAgent(session,
                     optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     summary_writer=writer)
