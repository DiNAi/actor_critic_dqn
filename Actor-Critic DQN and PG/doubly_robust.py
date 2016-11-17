import numpy as np
import tensorflow as tf

class DoublyRobust(object):
    def __init__(self, dqn_agent, behavior_policy, target_policy,
                    discount = 1):
        self.dqn_agent = dqn_agent
        self.behavior_policy = behavior_policy
        self.target_policy = target_policy
        self.discount = discount

    def computing_probabilities(self, episode):
        """Return probabilites of actions in present states for an EPISODE
        """
        probabilites = self.target_policy.compute_action_probabilities(episode["states"])
        return probabilites

    def computing_all_action_values(self, episode):
        """Return the action-values of the target policy for every actions and for all the states
        visited in a given EPISODE
        """
        q_values = self.dqn_agent.compute_all_q_values(episode["states"])
        return q_values

    def compute_state_values(self, episode):
        """Return the state-values for all the states in a given EPISODE
        """
        q_values = self.computing_all_action_values(episode)
        action_probabilities = self.computing_probabilities(episode)
        return np.sum(q_values * action_probabilities, axis=1)

    def compute_action_values(self, episode):
        """Return action values for the state and action visited in the given
        Episode
        """
        return self.dqn_agent.compute_q_values(episode["states"], episode["masked_actions"])

    def compute_action_probabilities(self, episode, policy):
        """Return the state-action probabilites for the given EPISODE
        and POLICY"""
        probs = policy.compute_action_probabilities(episode["states"])
        return probs[np.arange(len(probs)), episode["actions"]]

    def compute_importance_sampling_ratio(self, episode):
        """Return the importance sampling ratio between TARGET_POLICY and
        BEHAVIOR_POLICY"""
        target_policy_probs = self.compute_action_probabilities(episode, self.target_policy)
        behavior_policy_probs = self.compute_action_probabilities(episode, self.behavior_policy)
        return target_policy_probs / behavior_policy_probs

    def compute_episode(self, episode):
        """Return the doubly robust estimator for a given episode
        """
        doubly_robust_estimations = []
        v_dr = 0 # Doubly Robust Estimator
        action_values = self.compute_action_values(episode)
        state_values = self.compute_state_values(episode)
        importance_samplings = self.compute_importance_sampling_ratio(episode)
        rewards = episode["rewards"]
        for v, r, q, rho in zip(state_values, rewards, action_values, importance_samplings):
            v_dr = v + rho * (r + self.discount * v_dr - q)
            doubly_robust_estimations.append(v_dr)
        return doubly_robust_estimations[::-1] # dr estimates are calculated in reverse order

    def compute(self, episodes):
        """Return doubly robust estimator for the whole batch
        """
        return np.concatenate([self.compute_episode(episode) for episode in episodes])
