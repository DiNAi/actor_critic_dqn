import random
import numpy as np
import tensorflow as tf

class DQNAgent(object):

    def __init__(self, session,
                       optimizer,
                       q_network,
                       state_dim,
                       num_actions,
                       discount=0.9,
                       target_update_rate=0.01,
                       summary_writer=None,
                       summary_every=100):

        # tensorflow machinery
        self.session        = session
        self.optimizer      = optimizer
        self.summary_writer = summary_writer

        # model components
        self.q_network     = q_network

        # Q learning parameters
        self.state_dim          = state_dim
        self.num_actions        = num_actions
        self.discount    = discount
        self.target_update_rate = target_update_rate

        # counters
        self.training_itr = 0

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
        self.session.run(tf.initialize_variables(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every


    def create_input_placeholders(self):
        with tf.name_scope("inputs"):
            self.states = tf.placeholder(tf.float32, (None, self.state_dim), "states")
            self.actions = tf.placeholder(tf.float32, (None, self.num_actions), "actions")
            self.rewards = tf.placeholder(tf.float32, (None,), "rewards")
            self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), "next_stats")
            self.dones = tf.placeholder(tf.bool, (None,), "dones")
            self.next_action_probs = tf.placeholder(tf.float32, (None, self.num_actions), "next_action_probs")

    def create_variables_for_q_values(self):
        with tf.name_scope("action_values"):
            with tf.variable_scope("q_network"):
                self.q_values = self.q_network(self.states)
        with tf.name_scope("action_scores"):
            self.action_scores = tf.reduce_sum(tf.mul(self.q_values, self.actions), reduction_indices=1)

    def create_variables_for_target(self):
        with tf.name_scope("target_values"):
            with tf.variable_scope("target_network"):
                self.target_q_values = self.q_network(self.states)
            self.averaged_target_q_values = tf.reduce_sum(tf.mul(self.target_q_values,
                                                              self.next_action_probs),
                                                        reduction_indices=1)
            self.target_values = self.rewards + self.discount * self.averaged_target_q_values

    def create_variables_for_optimization(self):
        with tf.name_scope("optimization"):
            self.loss = tf.reduce_mean(tf.square(self.action_scores - self.target_values))
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(self.gradients)


    def create_summaries(self):
        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.histogram_summaries = []
        for grad, var in self.gradients:
            if grad is not None:
                histogram_summary = tf.histogram_summary(var.name + "/gradient", grad)
                self.histogram_summaries.append(histogram_summary)
        self.q_summary = tf.histogram_summary("q_summay", self.q_values)
        #self.episode_len = tf.scalar_summary("episode_len",)

    def merge_summaries(self):
        self.summarize = tf.merge_summary([self.loss_summary + self.q_summary]
                                           + self.histogram_summaries)


    def create_variables(self):
        self.create_input_placeholders()
        self.create_variables_for_q_values()
        self.create_variables_for_target()
        self.create_variables_for_optimization()


  # def create_variables_q_learning(self):
  #   # compute score of actions from a state
  #   with tf.name_scope("predict_actions_score"):
  #     # raw state representation
  #     self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")
  #     # initialize Q network
  #     with tf.variable_scope("q_network"):
  #       self.q_outputs = self.q_network(self.states)
  #     # predict actions from Q network
  #     self.action_scores = tf.identity(self.q_outputs, name="action_scores")
  #     #tf.histogram_summary("action_scores", self.action_scores)
  #
  #   # estimate rewards using the next state: r(s_t,a_t) + E_a Q(s_{t+1}, a)
  #   with tf.name_scope("estimate_future_rewards"):
  #     self.next_states = tf.placeholder(tf.float32, (None, self.state_dim), name="next_states")
  #     self.next_state_mask = tf.placeholder(tf.float32, (None,), name="next_state_masks")
  #     self.next_action_probabilities = tf.placeholder(tf.float32, (None, self.num_actions), name="next_action_probabilities")
  #     # initialize target network
  #     with tf.variable_scope("target_network"):
  #       self.target_outputs = self.q_network(self.next_states)
  #     # compute future rewards
  #     self.next_action_scores = tf.identity(self.target_outputs, name="next_action_scores")
  #     #tf.histogram_summary("next_action_scores", self.next_action_scores)
  #     next_state_rewards = tf.reduce_sum(self.next_action_probabilities * self.next_action_scores,
  #                                        reduction_indices=1, keep_dims=True) * self.next_state_mask
  #     next_state_rewards = tf.stop_gradient(next_state_rewards)
  #     self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
  #     self.future_rewards = self.rewards + self.discount_factor * next_state_rewards
  #
  #   # compute loss and gradients
  #   with tf.name_scope("compute_temporal_differences"):
  #     # compute temporal difference loss
  #     self.action_mask = tf.placeholder(tf.float32, (None, self.num_actions), name="action_mask")
  #     self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1,])
  #     self.temp_diff = self.masked_action_scores - self.future_rewards
  #     self.td_loss = tf.reduce_mean(tf.square(self.temp_diff))
  #     # regularization loss
  #     q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
  #     self.reg_loss = self.reg_param * tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in q_network_variables])
  #     # compute total loss and gradients
  #     self.loss = self.td_loss + self.reg_loss
  #     gradients = self.optimizer.compute_gradients(self.loss)
  #     # clip gradients by norm
  #     for i, (grad, var) in enumerate(gradients):
  #       if grad is not None:
  #         gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)
  #     # add histograms for gradients.
  #     # for grad, var in gradients:
  #       #tf.histogram_summary(var.name, var)
  #       #if grad is not None:
  #       #  tf.histogram_summary(var.name + '/gradients', grad)
  #     self.train_op = self.optimizer.apply_gradients(gradients)
  #
  #   # update target network with Q network
  #   with tf.name_scope("update_target_network"):
  #     self.target_network_update = []
  #     # slowly update target network parameters with Q network parameters
  #     q_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
  #     target_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_network")
  #     for v_source, v_target in zip(q_network_variables, target_network_variables):
  #       # this is equivalent to target = (1-alpha) * target + alpha * source
  #       update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
  #       self.target_network_update.append(update_op)
  #     self.target_network_update = tf.group(*self.target_network_update)
  #
  #   # scalar summaries
  #   tf.scalar_summary("td_loss", self.td_loss)
  #   tf.scalar_summary("reg_loss", self.reg_loss)
  #   tf.scalar_summary("total_loss", self.loss)
  #
  #   self.summarize = tf.merge_all_summaries()
  #   self.no_op = tf.no_op()
  #
  # def q_values(self, states, actions):
  #   output = self.session.run(self.q_outputs, feed_dict={self.states:states})
  #   return(output[np.arange(states.shape[0]), actions]) # return q_values for only actions taken in a given state
  #
  # def storeExperience(self, states, actions, rewards, next_states, next_action_probs, dones):
  #   # always store end states
  #   for (state, action, reward, next_state, next_action_probs, done) in zip(states, actions, rewards, next_states, next_action_probs, dones):
  #     if self.store_experience_cnt == 0 or done:
  #       self.replay_buffer.add(state, action, reward, next_state, next_action_probs, done)
  #     self.store_experience_cnt = (self.store_experience_cnt + 1) % self.store_replay_every
  #
  # def updateModel(self):
  #   # not enough experiences yet
  #   if self.replay_buffer.count() < self.batch_size:
  #     return
  #
  #   batch           = self.replay_buffer.getBatch(self.batch_size)
  #   states          = np.zeros((self.batch_size, self.state_dim))
  #   rewards         = np.zeros((self.batch_size,))
  #   action_mask     = np.zeros((self.batch_size, self.num_actions))
  #   next_states     = np.zeros((self.batch_size, self.state_dim))
  #   next_state_mask = np.zeros((self.batch_size,))
  #   next_action_probabilities = np.zeros((self.batch_size, self.num_actions))
  #
  #   for k, (s0, a, r, s1, a1, done) in enumerate(batch):
  #     states[k] = s0
  #     rewards[k] = r
  #     action_mask[k][a] = 1
  #     next_action_probabilities[k] = a1
  #     # check terminal state
  #     if not done:
  #       next_states[k] = s1
  #       next_state_mask[k] = 1
  #
  #   # whether to calculate summaries
  #   calculate_summaries = self.train_iteration % self.summary_every == 0 and self.summary_writer is not None
  #
  #   # perform one update of training
  #   cost, _ = self.session.run([ # remove summary_str
  #     self.loss,
  #     self.train_op,
  #     #self.summarize if calculate_summaries else self.no_op
  #   ], {
  #     self.states:          states,
  #     self.next_states:     next_states,
  #     self.next_state_mask: next_state_mask,
  #     self.action_mask:     action_mask,
  #     self.rewards:         rewards,
  #     self.next_action_probabilities: next_action_probabilities
  #   })
  #
  #
  #   # update target network using Q-network
  #   self.session.run(self.target_network_update)
  #
  #   # emit summaries
  #   # if calculate_summaries:
  #   #   self.summary_writer.add_summary(summary_str, self.train_iteration)
  #
  #   self.train_iteration += 1
