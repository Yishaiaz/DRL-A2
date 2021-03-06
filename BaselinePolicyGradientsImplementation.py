import itertools

import gym
import os
import shutil
import numpy as np
import tensorflow as tf2
import collections
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

env = gym.make('CartPole-v1')

np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.weights_initializer = tf.keras.initializers.glorot_normal(seed=0)

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=self.weights_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=self.weights_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# Metrics:
class CustomMetric(tf.keras.metrics.Metric):
    """
    custom metric to keep track of loss in each training step
    """
    def __init__(self, name='training_step_loss', dtype=tf.int32):
        super(CustomMetric, self).__init__(name=name, dtype=dtype)
        self.val = None

    def update_state(self, x):
        self.val = x

    def result(self):
        return self.val


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 2000
max_steps = 501

# Define hyperparameters
discount_factors = (0.99, 0.9)
learning_rates = (0.0004, 0.001)

for learning_rate, discount_factor in itertools.product(learning_rates, discount_factors):
    parameters_dict = {'lr': learning_rate, 'discount': discount_factor}

    # TENSORBOARD
    exp_details = '_'.join([f'{key}={val}' for key, val in parameters_dict.items()])
    parameters_dict = {'lr': learning_rate, 'discount': discount_factor}

    # TENSORBOARD
    exp_details = '_'.join([f'{key}={val}' for key, val in parameters_dict.items()])
    exp_details = f'BaselinePolicy_{exp_details}'

    main_dir_to_save = os.sep.join([os.getcwd(), 'Experiments'])
    exp_dir_to_save_train = os.sep.join([main_dir_to_save, exp_details, 'train'])
    exp_dir_to_save_test = os.sep.join([main_dir_to_save, exp_details, 'test'])
    # remove all existing dirs and files with the same experiment identifier.
    if os.path.isdir(exp_dir_to_save_train):
        shutil.rmtree(exp_dir_to_save_train, ignore_errors=True)
    if os.path.isdir(exp_dir_to_save_test):
        shutil.rmtree(exp_dir_to_save_test, ignore_errors=True)

    render = False

    # Initialize the policy network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)

    tf.compat.v1.summary.scalar(name="episode_reward", tensor=policy.R_t)
    tf.compat.v1.summary.scalar(name="episode_learning_rate", tensor=policy.learning_rate)
    tf.compat.v1.summary.scalar(name="episode_loss", tensor=policy.loss)

    tfb_train_summary_writer = tf.summary.FileWriter(exp_dir_to_save_train)
    summaries = tf.compat.v1.summary.merge_all()

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
                feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
                _, loss = sess.run([policy.optimizer, policy.loss], feed_dict)

                feed_dict[policy.R_t] = episode_rewards[episode]
                summary = sess.run(summaries, feed_dict)
                tfb_train_summary_writer.add_summary(summary, episode)
