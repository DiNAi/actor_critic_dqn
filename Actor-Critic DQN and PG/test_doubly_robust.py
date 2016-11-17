import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
from dqn_agent import DQNAgent
from sampler import Sampler
from doubly_robust import DoublyRobust

env = gym.make("CartPole-v0")

class FixedPolicy(object):
    def __init__(self):
        pass

    def sampleAction(self, state):
        # Always return action "0"
        return np.random.choice([0, 1])

    def compute_action_probabilities(self, states):
        # return the probabilites for the states
        return np.array([[0.5, 0.5]] * len(states))

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

session = tf.Session()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

fixed_policy = FixedPolicy()
sampler = Sampler(fixed_policy, env, discount=1, use_doubly_robust=True)

def action_masker(array):
    masked_action = np.zeros((array.size, num_actions))
    masked_action[np.arange(array.size), array] = 1
    return masked_action

def probs_for_next_action(array):
    return np.column_stack((0.5 * np.ones_like(array), 0.5 * np.ones_like(array)))

dqn_agent = DQNAgent(session,
                     optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     summary_writer=writer)

dr = DoublyRobust(dqn_agent, fixed_policy, fixed_policy)

episodes, batch = sampler.collect_one_batch()

for episode in episodes:
    masked_action = action_masker(np.array(episode["actions"]))
    episode["masked_actions"] = masked_action
