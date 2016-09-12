import unittest
import gym
import numpy as np
import tensorflow as tf
import sampler
from replay_buffer import ReplayBuffer
from pg_reinforce import PolicyGradientREINFORCE

env = gym.make("CartPole-v0")

class FixedPolicy(object):
    def __init__(self):
        pass

    def sampleAction(self, state):
        # Always return action "0"
        return 0

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

fixed_policy = FixedPolicy()
sampler = sampler.Sampler(fixed_policy, env, discount=1)

sess = tf.Session()
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
writer = tf.train.SummaryWriter("/home/drl/DRL/tensorflow-reinforce/tmp/")

pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       summary_writer=writer)

class SamplerTests(unittest.TestCase):
    def testMonteCarloReturns(self):
        return_true = sampler.compute_monte_carlo_returns([1, 2, 3])
        return_false = sampler.compute_monte_carlo_returns([1, 2, 10])
        self.assertEqual([6,5,3], return_true)
        self.assertNotEqual([13, 12, 9], return_false)

    def testCollectOneEpisode(self):
        episode = sampler.collect_one_episode()
        states = episode["states"]
        actions = episode["actions"]
        rewards = episode["rewards"]
        returns = episode["monte_carlo_returns"]
        next_states = episode["next_states"]
        dones = episode["dones"]
        monte_carlo_returns = np.cumsum(rewards[::-1])[::-1].tolist()
        # checking some immutability of an episode
        self.assertEqual(states[1:], next_states[:-1])
        self.assertFalse(any(actions))
        self.assertEqual(returns, monte_carlo_returns)
        self.assertTrue(dones[-1])
        self.assertFalse(any(dones[:-1]))

    def testCollectOneBatch(self):
        episodes = sampler.collect_one_batch()

class PolicyGradientTest(unittest.TestCase):

    def testComputeAction(self):
        observation = env.observation_space.sample()
        observation = observation[np.newaxis, :]
        self.assertTrue(pg_reinforce.sampleAction(observation) in range(0, num_actions))

    def testComputeActionsProbs(self):
        observations = []
        for _ in range(10):
            observations.append(env.observation_space.sample())
        observations = np.array(observations)
        probs = pg_reinforce.compute_action_probabilities(observations)
        self.assertTrue(probs.shape == (observations.shape[0], num_actions))

    def testComputeGradient(self):
        import sampler
        sampler = sampler.Sampler(pg_reinforce, env)
        batch = sampler.collect_one_batch()
        grads = pg_reinforce.session.run(pg_reinforce.gradients[0][0], {pg_reinforce.states : batch["states"],
                                                                        pg_reinforce.actions : batch["actions"],
                                                                        pg_reinforce.returns : batch["monte_carlo_returns"]} )
        self.assertTrue(grads.shape == (4, 20))

    def testComputeNewParamemters(self):
        import sampler
        sampler = sampler.Sampler(pg_reinforce, env)
        batch = sampler.collect_one_batch()
        variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
        prev_val = pg_reinforce.session.run(variable)
        grad = pg_reinforce.session.run(pg_reinforce.gradients[0][0], {pg_reinforce.states : batch["states"],
                                                                        pg_reinforce.actions : batch["actions"],
                                                                        pg_reinforce.returns : batch["monte_carlo_returns"]} )
        pg_reinforce.session.run(pg_reinforce.train_op, {pg_reinforce.states : batch["states"],
                                                         pg_reinforce.actions : batch["actions"],
                                                         pg_reinforce.returns : batch["monte_carlo_returns"]} )
        post_val = pg_reinforce.session.run(variable)
        diff = post_val - prev_val + grad
        self.assertTrue(np.allclose(diff, 0, atol=1e-6))

    def testStandAloneConvergence(self):
        import sampler
        from tqdm import tqdm
        NUM_ITR = 5
        sampler = sampler.Sampler(pg_reinforce, env)
        for _ in tqdm(range(NUM_ITR)):
            batch = sampler.collect_one_batch()
            grads = pg_reinforce.update_parameters(batch["states"], batch["actions"], batch["monte_carlo_returns"])
        self.assertTrue(True)

class ReplayBufferTest(unittest.TestCase):
    def testStore(self):
        replay_buffer = ReplayBuffer(3)
        replay_buffer.add(1)
        self.assertTrue(list(replay_buffer.buffer) == [1])

    def testDiscardOldElement(self):
        replay_buffer = ReplayBuffer(3)
        replay_buffer.add_items([2, 3, 4, 5])
        self.assertTrue(replay_buffer.buffer.popleft() != 1)

    def testRandomSample(self):
        replay_buffer = ReplayBuffer(3)
        replay_buffer.add_items([10, 12, 14])
        sample = replay_buffer.sample(2)
        self.assertTrue(len(sample) == 2)

class DQNAgentTest(unittest.TestCase):
    def testActionValues(self):
        self.assertTrue(False)

    def testUpdate(self):
        self.assertTrue(False)

    def testLearn(self):
        self.assertTrue(False)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
