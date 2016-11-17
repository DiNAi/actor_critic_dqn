import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from pg_reinforce import PolicyGradientREINFORCE
from replay_buffer import ReplayBuffer
from sampler import Sampler
from doubly_robust import DoublyRobust

# Environment parameters
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim   = env.observation_space.shape[0]
num_actions = env.action_space.n

# Policy nework parameters
entropy_bonus = 0.5
policy_session = tf.Session()
policy_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
policy_writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/policy/")
policy_summary_every = 10

def show_image(array):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(array)
    plt.title("Reward Progress")
    plt.xlabel("Iteration number")
    plt.ylabel("rewards")
    plt.grid()
    plt.show()

def policy_network(states):
   """ define policy neural network """
   W1 = tf.get_variable("W1", [state_dim, 20],
                        initializer=tf.truncated_normal_initializer())
   b1 = tf.get_variable("b1", [20],
                        initializer=tf.constant_initializer(0))
   h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
   W2 = tf.get_variable("W2", [20, num_actions],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
   b2 = tf.get_variable("b2", [num_actions],
                        initializer=tf.constant_initializer(0))
   p = tf.matmul(h1, W2) + b2
   return p

pg_reinforce = PolicyGradientREINFORCE(policy_session,
                                       policy_optimizer,
                                       policy_network,
                                       state_dim,
                                       entropy_bonus=entropy_bonus,
                                       summary_writer=policy_writer,
                                       summary_every=policy_summary_every)

# Initializing Sampler
sampler = Sampler(pg_reinforce, env, use_doubly_robust=True)

# Q-network parameters
q_session = tf.Session()
q_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
q_writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/q/")
q_summary_every = 10

def action_masker(array):
    masked_action = np.zeros((array.size, num_actions), dtype=np.float32)
    masked_action[np.arange(array.size), array] = 1.0
    return masked_action

def q_network(states):
    W1 = tf.get_variable("W1", [state_dim, 20],
                         initializer=tf.truncated_normal_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.truncated_normal_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                         initializer=tf.constant_initializer(0))
    q = tf.matmul(h1, W2) + b2
    return q

dqn_agent = DQNAgent(q_session,
                     q_optimizer,
                     q_network,
                     state_dim,
                     num_actions,
                     target_update_rate=0.01,
                     summary_writer=q_writer,
                     summary_every=q_summary_every)

# Initializing ReplayBuffer
buffer_size = 100000
sample_size = 2**13
replay_buffer = ReplayBuffer(buffer_size)

# Training
def computing_probabilities(batch):
    probabilites = pg_reinforce.compute_action_probabilities(batch["next_states"])
    return probabilites

def update_batch(batch):
    masked_action = action_masker(batch["actions"])
    batch["actions"] = masked_action

def update_random_batch(batch):
    next_action_probs = computing_probabilities(batch)
    batch["next_action_probs"] = next_action_probs

def update_q_parameters(batch):
    dqn_agent.update_parameters(batch)

def compute_return(batch):
    return dqn_agent.compute_q_values(batch["states"], batch["actions"])

def update_episodes(episodes):
    for episode in episodes:
        masked_action = action_masker(np.array(episode["actions"]))
        episode["masked_actions"] = masked_action

doubly_robust_estimator = DoublyRobust(dqn_agent, pg_reinforce, pg_reinforce)

reward = []
for _ in trange(1000):
    episodes, batch = sampler.collect_one_batch()
    actions = batch["actions"]
    update_batch(batch)
    replay_buffer.add_batch(batch)
    if sample_size <= replay_buffer.num_items:
        random_batch = replay_buffer.sample_batch(sample_size)
        update_random_batch(random_batch)
        update_q_parameters(random_batch)
        returns = doubly_robust_estimator.compute(episodes)
        pg_reinforce.update_parameters(batch["states"], actions, returns)
        reward.append(batch["rewards"].sum()/200)

show_image(reward)
