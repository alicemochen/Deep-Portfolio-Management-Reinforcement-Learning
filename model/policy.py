import os

import tensorflow.compat.v1 as tf
import numpy as np

# define neural net \pi_\phi(s) as a class
from model.pvm import PVM
from utils.utils import get_random_action, eval_perf


class Policy(object):
    '''
    This class is used to instanciate the policy network agent

    '''

    def __init__(self, sess, env, config):
        # parameters
        self.config = config
        self.env = env
        self.sess = sess
        self.trading_cost = config["dict_fin"]["trading_cost"]
        self.interest_rate = config["dict_fin"]["interest_rate"]
        self.n_filter_1 = config['dict_hp_net']['n_filter_1']
        self.n_filter_2 = config['dict_hp_net']['n_filter_2']
        # n is number of trading period
        self.length_tensor = config['dict_hp_pb']['length_tensor']
        # m is number of stocks
        self.nb_stocks = config["nb_stocks"]
        self.input_dir = config["input_dir"]

        with tf.variable_scope("Inputs"):
            # Placeholder

            # tensor of the prices
            self.X_t = tf.placeholder(
                tf.float32, [None, config["nb_feature_map"], self.nb_stocks, self.length_tensor])  # The Price tensor
            # weights at the previous time step
            self.W_previous = tf.placeholder(tf.float32, [None, self.nb_stocks + 1])
            # portfolio value at the previous time step
            self.pf_value_previous = tf.placeholder(tf.float32, [None, 1])
            # vector of Open(t+1)/Open(t)
            self.dailyReturn_t = tf.placeholder(tf.float32, [None, self.nb_stocks])

            # self.pf_value_previous_eq = tf.placeholder(tf.float32, [None, 1])

        with tf.variable_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.config['dict_hp_opt']['learning'])

        with tf.variable_scope("Policy_Model"):
            # variable of the cash bias
            bias = tf.get_variable('cash_bias', shape=[
                1, 1, 1, 1], initializer=tf.constant_initializer(self.config["dict_fin"]["cash_bias_init"]))
            # shape of the tensor == batchsize
            shape_X_t = tf.shape(self.X_t)[0]
            # trick to get a "tensor size" for the cash bias
            self.cash_bias = tf.tile(bias, tf.stack([shape_X_t, 1, 1, 1]))
            # print(self.cash_bias.shape)

            with tf.variable_scope("Conv1"):
                # first layer on the X_t tensor
                # return a tensor of depth 2
                self.conv1 = tf.layers.conv2d(
                    inputs=tf.transpose(self.X_t, perm=[0, 3, 2, 1]),
                    activation=tf.nn.relu,
                    filters=self.n_filter_1,
                    strides=(1, 1),
                    kernel_size=self.config["dict_hp_net"]["kernel1_size"],
                    padding='same')

            with tf.variable_scope("Conv2"):
                # feature maps
                self.conv2 = tf.layers.conv2d(
                    inputs=self.conv1,
                    activation=tf.nn.relu,
                    filters=self.n_filter_2,
                    strides=(self.length_tensor, 1),
                    kernel_size=(1, self.length_tensor),
                    padding='same')

            with tf.variable_scope("Tensor3"):
                # w from last periods
                # trick to have good dimensions
                w_wo_c = self.W_previous[:, 1:]
                w_wo_c = tf.expand_dims(w_wo_c, 1)
                w_wo_c = tf.expand_dims(w_wo_c, -1)
                self.tensor3 = tf.concat([self.conv2, w_wo_c], axis=3)

            with tf.variable_scope("Conv3"):
                # last feature map WITHOUT cash bias
                self.conv3 = tf.layers.conv2d(
                    inputs=self.conv2,
                    activation=tf.nn.relu,
                    filters=1,
                    strides=(self.n_filter_2 + 1, 1),
                    kernel_size=(1, 1),
                    padding='same')

            with tf.variable_scope("Tensor4"):
                # last feature map WITH cash bias
                self.tensor4 = tf.concat([self.cash_bias, self.conv3], axis=2)
                # we squeeze to reduce and get the good dimension
                self.squeezed_tensor4 = tf.squeeze(self.tensor4, [1, 3])

            with tf.variable_scope("Policy_Output"):
                # softmax layer to obtain weights
                self.action = tf.nn.softmax(self.squeezed_tensor4)

            with tf.variable_scope("Reward"):
                # computation of the reward
                # please look at the chronological map to understand
                constant_return = tf.constant(
                    1 + self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)
                Vprime_t = self.action * self.pf_value_previous
                Vprevious = self.W_previous * self.pf_value_previous

                # this is just a trick to get the good shape for cost
                constant = tf.constant(1.0, shape=[1])

                cost = self.trading_cost * \
                       tf.norm(Vprime_t - Vprevious, ord=1, axis=1) * constant

                cost = tf.expand_dims(cost, 1)

                zero = tf.constant(
                    np.array([0.0] * self.nb_stocks).reshape(1, self.nb_stocks), shape=[1, self.nb_stocks], dtype=tf.float32)

                vec_zero = tf.tile(zero, tf.stack([shape_X_t, 1]))
                vec_cost = tf.concat([cost, vec_zero], axis=1)

                Vsecond_t = Vprime_t - vec_cost

                V_t = tf.multiply(Vsecond_t, y_t)
                self.portfolioValue = tf.norm(V_t, ord=1)
                self.instantaneous_reward = (self.portfolioValue - self.pf_value_previous) / self.pf_value_previous

            with tf.variable_scope("Reward_Equiweighted"):
                constant_return = tf.constant(
                    1 + self.interest_rate, shape=[1, 1])
                cash_return = tf.tile(
                    constant_return, tf.stack([shape_X_t, 1]))
                y_t = tf.concat(
                    [cash_return, self.dailyReturn_t], axis=1)

                w_eq = np.array(np.array([1 / (self.nb_stocks + 1)] * (self.nb_stocks + 1)))
                V_eq = w_eq * self.pf_value_previous
                V_eq_second = tf.multiply(V_eq, y_t)

                self.portfolioValue_eq = tf.norm(V_eq_second, ord=1)

                self.instantaneous_reward_eq = (self.portfolioValue_eq - self.pf_value_previous) / self.pf_value_previous

            with tf.variable_scope("Max_weight"):
                self.max_weight = tf.reduce_max(self.action)
                print(self.max_weight.shape)

            with tf.variable_scope("Reward_adjusted"):
                self.adjusted_reward = self.instantaneous_reward - self.instantaneous_reward_eq - self.config["dict_hp_pb"]["ratio_regul"] * self.max_weight

        # objective function
        # maximize reward over the batch
        # min(-r) = max(r)
        self.train_op = self.optimizer.minimize(-self.adjusted_reward)


    def compute_W(self, X_t_, W_previous_):
        """
        This function returns the action the agent takes
        given the input tensor and the W_previous

        It is a vector of weight

        """

        return self.sess.run(tf.squeeze(self.action), feed_dict={self.X_t: X_t_, self.W_previous: W_previous_})

    def train(self):
        """
        This function trains the neural network
        maximizing the reward
        the input is a batch of the differents values
        """
        train_save_dir = f"{self.input_dir}\\train_graphs"
        if not os.path.exists(train_save_dir):
            os.mkdir(train_save_dir)
        list_final_pf = list()
        total_steps_train = int(self.config['dict_hp_pb']['ratio_train'] * self.config["trading_period"])
        w_init_train = np.array(np.array([1] + [0] * self.config["nb_stocks"]))
        for e in range(self.config["dict_train"]["n_episodes"]):
            print('Episode:', e)
            # init the PVM with the training parameters
            memory = PVM(self.config, total_steps_train, w_init_train)

            for nb in range(self.config["dict_train"]["n_batches"]):
                # draw the starting point of the batch
                i_start = memory.draw()

                # reset the environment with the weight from PVM at the starting point
                # reset also with a portfolio value with initial portfolio value
                state, done = self.env.reset(memory.get_W(i_start), self.config['dict_train']['pf_init_train'], t=i_start)

                list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t = [], [], [], []

                for bs in range(self.config["dict_hp_pb"]["batch_size"]):

                    # load the different inputs from the previous loaded state
                    X_t = state[0].reshape([-1] + list(state[0].shape))
                    W_previous = state[1].reshape([-1] + list(state[1].shape))
                    pf_value_previous = state[2]

                    if np.random.rand() < self.config["dict_hp_pb"]["ratio_greedy"]:
                        # print('go')
                        # computation of the action of the agent
                        action = self.compute_W(X_t, W_previous)
                    else:
                        action = get_random_action(self.config["nb_stocks"])

                    # given the state and the action, call the environment to go one time step later
                    state, reward, done = self.env.step(action)

                    # get the new state
                    X_next = state[0]
                    W_t = state[1]
                    pf_value_t = state[2]

                    # let us compute the returns
                    dailyReturn_t = X_next[-1, :, -1]
                    # update into the PVM
                    memory.update(i_start + bs, W_t)
                    # store elements
                    list_X_t.append(X_t.reshape(state[0].shape))
                    list_W_previous.append(W_previous.reshape(state[1].shape))
                    list_pf_value_previous.append([pf_value_previous])
                    list_dailyReturn_t.append(dailyReturn_t)

                    if bs == self.config["dict_hp_pb"]["batch_size"] - 1:
                        list_final_pf.append(pf_value_t)

                list_W_previous = np.array(list_W_previous)
                list_pf_value_previous = np.array(list_pf_value_previous)
                list_dailyReturn_t = np.array(list_dailyReturn_t)
                # for each batch, train the network to maximize the reward
                self.sess.run(self.train_op, feed_dict={self.X_t: list_X_t,
                                                        self.W_previous: list_W_previous,
                                                        self.pf_value_previous: list_pf_value_previous,
                                                        self.dailyReturn_t: list_dailyReturn_t})

            # eval_perf(e, data, actor, total_steps_train, total_steps_val, w_init_eval, config, train_save_dir)

