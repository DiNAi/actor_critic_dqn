import tensorflow as tf
import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pg_reinforce import PolicyGradientREINFORCE
from sampler import Sampler

env = gym.make("CartPole-v0")
sess = tf.Session()
state_dim = 4
num_actions = 2
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

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

pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       summary_writer=writer)

sampler = Sampler(pg_reinforce, env)

reward = []
for _ in tqdm(range(30)):
    batch = sampler.collect_one_batch()
    pg_reinforce.update_parameters(batch["states"], batch["actions"], batch["monte_carlo_returns"])
    reward.append(batch["rewards"].sum()/200)

show_image(reward)


# batch = sampler.collect_one_batch()
# variables = tf.get_collection(tf.GraphKeys.VARIABLES)
# trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# variable_names = '\n'.join([v.name for v in variables])
# trainable_variable_names = '\n'.join([v.name for v in trainable_variables])
#
# prev_val = pg_reinforce.session.run(variables[0])
# grad1 = pg_reinforce.session.run(pg_reinforce.gradients[0][0], {pg_reinforce.states : batch["states"],
#                                                                 pg_reinforce.actions : batch["actions"],
#                                                                 pg_reinforce.returns : batch["monte_carlo_returns"]} )
# pg_reinforce.session.run(pg_reinforce.train_op, {pg_reinforce.states : batch["states"],
#                                                                 pg_reinforce.actions : batch["actions"],
#                                                                 pg_reinforce.returns : batch["monte_carlo_returns"]} )
#
# post_val = pg_reinforce.session.run(variables[0])
# print "variable_names: {}".format(variable_names)
# print "trainable_variables: {}".format(trainable_variable_names)
# diff = (post_val - prev_val + 1.0*grad1)
# # print diff[np.logical_not(np.isclose(diff, 0, 1e-4))]
# # print diff[np.isclose(diff, 0, 1e-4)]
# print np.isclose(diff, 0, atol=1e-6)
# print diff
# # pg_reinforce.create_variables()
# pg_reinforce.summary_writer.add_graph(pg_reinforce.session.graph)
