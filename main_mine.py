from data_utils import get_stock_features_default
from utils import eval_perf, get_random_action
from policy import Policy
import tensorflow.compat.v1 as tf
from pvm import PVM
from environment import TradeEnv
import os
import json
import numpy as np

ROOT_DIR = os.path.abspath(os.curdir)


def train_actor(config, sess, optimizer, input_dir):
    list_stock = config["names"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    interest_rate = config["dict_fin"]["interest_rate"]
    trading_cost = config["dict_fin"]["trading_cost"]
    data = get_stock_features_default(list_stock, start_date, end_date)
    trading_period = data.shape[2]
    nb_feature_map = data.shape[0]
    nb_stocks = data.shape[1]

    # Total number of steps for pre-training in the training set
    total_steps_train = int(config['dict_hp_pb']['ratio_train'] * trading_period)
    total_steps_val = int(config['dict_hp_pb']['ratio_val'] * trading_period)
    total_steps_test = trading_period - total_steps_train - total_steps_val

    w_init_train = np.array(np.array([1] + [0] * nb_stocks))
    w_init_eval = np.array(np.array([1] + [0] * nb_stocks))

    # initialize networks
    length_tensor = config['dict_hp_pb']['length_tensor']
    pf_init_train = config['dict_train']['pf_init_train']
    actor = Policy(nb_stocks, sess, optimizer, nb_feature_map, config)  # policy initialization
    env = TradeEnv(data, window_length=length_tensor,
                   portfolio_value=pf_init_train, trading_cost=trading_cost,
                   interest_rate=interest_rate, train_size=config['dict_hp_pb']['ratio_train'])
    sess.run(tf.global_variables_initializer())
    list_final_pf = list()

    n_episodes = config["dict_train"]["n_episodes"]
    n_batches = config["dict_train"]["n_batches"]
    sample_bias = config["sample_bias"]
    batch_size = config["dict_hp_pb"]["batch_size"]
    ratio_greedy = config["dict_hp_pb"]["ratio_greedy"]
    for e in range(n_episodes):
        print('Episode:', e)
        # init the PVM with the training parameters
        memory = PVM(nb_stocks, sample_bias, total_steps_train, w_init_train,
                     batch_size=batch_size)

        for nb in range(n_batches):
            # draw the starting point of the batch
            i_start = memory.draw()

            # reset the environment with the weight from PVM at the starting point
            # reset also with a portfolio value with initial portfolio value
            state, done = env.reset(memory.get_W(i_start), pf_init_train, t=i_start)

            list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t = [], [], [], []

            for bs in range(batch_size):

                # load the different inputs from the previous loaded state
                X_t = state[0].reshape([-1] + list(state[0].shape))
                W_previous = state[1].reshape([-1] + list(state[1].shape))
                pf_value_previous = state[2]

                if np.random.rand() < ratio_greedy:
                    # print('go')
                    # computation of the action of the agent
                    action = actor.compute_W(X_t, W_previous)
                else:
                    action = get_random_action(nb_stocks)

                # given the state and the action, call the environment to go one time step later
                state, reward, done = env.step(action)

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

                if bs == batch_size - 1:
                    list_final_pf.append(pf_value_t)

            list_W_previous = np.array(list_W_previous)
            list_pf_value_previous = np.array(list_pf_value_previous)
            list_dailyReturn_t = np.array(list_dailyReturn_t)
            # for each batch, train the network to maximize the reward
            actor.train(list_X_t, list_W_previous,
                        list_pf_value_previous, list_dailyReturn_t)
        train_save_dir = f"{input_dir}\\train_graphs"
        if not os.path.exists(train_save_dir):
            os.mkdir(train_save_dir)
        eval_perf(e, data, actor, total_steps_train, total_steps_val, w_init_eval, train_save_dir)


if __name__ == "__main__":

    input_dirs = ['run_directory\\tech_stocks']
    tf.reset_default_graph()
    tf.disable_eager_execution()
    sess = tf.Session()
    for input_dir in input_dirs:
        config = json.load(open(f"{ROOT_DIR}\\{input_dir}\\config.json"))
        learning = config['dict_hp_opt']['learning']
        optimizer = tf.train.AdamOptimizer(learning)
        train_actor(config, sess, optimizer, input_dir)

    # w_eq = np.array(np.array([1 / (nb_stocks + 1)] * (nb_stocks + 1)))
    # w_s = np.array(np.array([1] + [0.0] * nb_stocks))

    #
    # env, env_eq, env_s = create_environment(data)

    # full on one stock environment
    # these environments are set up for agents who play only on one stock
    # action_fu = list()
    # env_fu = list()
    # for i in range(nb_stocks):
    #     action = np.array([0] * (i + 1) + [1] + [0] * (nb_stocks - (i + 1)))
    #     action_fu.append(action)
    #
    #     env_fu_i = TradeEnv(data, window_length=n,
    #                         portfolio_value=pf_init_train, trading_cost=trading_cost,
    #                         interest_rate=interest_rate, train_size=dict_hp_pb['ratio_train'])
    #
    #     env_fu.append(env_fu_i)

    # initialize tensorflow graphs

    # list_final_pf_eq = list()
    # list_final_pf_s = list()

    # list_final_pf_fu = list()
    # state_fu = [0] * nb_stocks
    # done_fu = [0] * nb_stocks

    # pf_value_t_fu = [0] * nb_stocks

    # for i in range(nb_stocks):
    #     list_final_pf_fu.append(list())

    ###### Train #####
    # for e in range(n_episodes):
    #     print('Start Episode', e)
    #     # if e == 0:
    #     #     eval_perf('Before Training', actor)
    #     print('Episode:', e)
    #     # init the PVM with the training parameters
    #     memory = PVM(nb_stocks, sample_bias, total_steps_train, w_init_train,
    #                  batch_size=batch_size)
    #
    #     for nb in range(n_batches):
    #         # draw the starting point of the batch
    #         i_start = memory.draw()
    #
    #         # reset the environment with the weight from PVM at the starting point
    #         # reset also with a portfolio value with initial portfolio value
    #         state, done = env.reset(memory.get_W(i_start), pf_init_train, t=i_start)
    #         # state_eq, done_eq = env_eq.reset(w_eq, pf_init_train, t=i_start)
    #         # state_s, done_s = env_s.reset(w_s, pf_init_train, t=i_start)
    #
    #         # for i in range(nb_stocks):
    #         #     state_fu[i], done_fu[i] = env_fu[i].reset(action_fu[i], pf_init_train, t=i_start)
    #
    #         list_X_t, list_W_previous, list_pf_value_previous, list_dailyReturn_t = [], [], [], []
    #         # list_pf_value_previous_eq, list_pf_value_previous_s = [], []
    #         # list_pf_value_previous_fu = list()
    #         # for i in range(nb_stocks):
    #         #     list_pf_value_previous_fu.append(list())
    #
    #         for bs in range(batch_size):
    #
    #             # load the different inputs from the previous loaded state
    #             X_t = state[0].reshape([-1] + list(state[0].shape))
    #             W_previous = state[1].reshape([-1] + list(state[1].shape))
    #             pf_value_previous = state[2]
    #
    #             if np.random.rand() < ratio_greedy:
    #                 # print('go')
    #                 # computation of the action of the agent
    #                 action = actor.compute_W(X_t, W_previous)
    #             else:
    #                 action = get_random_action(nb_stocks)
    #
    #             # given the state and the action, call the environment to go one time step later
    #             state, reward, done = env.step(action)
    #             # state_eq, reward_eq, done_eq = env_eq.step(w_eq)
    #             # state_s, reward_s, done_s = env_s.step(w_s)
    #             # for i in range(nb_stocks):
    #             #     state_fu[i], _, done_fu[i] = env_fu[i].step(action_fu[i])
    #
    #             # get the new state
    #             X_next = state[0]
    #             W_t = state[1]
    #             pf_value_t = state[2]
    #
    #             # pf_value_t_eq = state_eq[2]
    #             # pf_value_t_s = state_s[2]
    #
    #             # for i in range(nb_stocks):
    #             #     pf_value_t_fu[i] = state_fu[i][2]
    #
    #             # let us compute the returns
    #             dailyReturn_t = X_next[-1, :, -1]
    #             # update into the PVM
    #             memory.update(i_start + bs, W_t)
    #             # store elements
    #             list_X_t.append(X_t.reshape(state[0].shape))
    #             list_W_previous.append(W_previous.reshape(state[1].shape))
    #             list_pf_value_previous.append([pf_value_previous])
    #             list_dailyReturn_t.append(dailyReturn_t)
    #
    #             # list_pf_value_previous_eq.append(pf_value_t_eq)
    #             # list_pf_value_previous_s.append(pf_value_t_s)
    #
    #             # for i in range(nb_stocks):
    #             #     list_pf_value_previous_fu[i].append(pf_value_t_fu[i])
    #
    #             if bs == batch_size - 1:
    #                 list_final_pf.append(pf_value_t)
    #                 # list_final_pf_eq.append(pf_value_t_eq)
    #                 # list_final_pf_s.append(pf_value_t_s)
    #                 # for i in range(nb_stocks):
    #                 #     list_final_pf_fu[i].append(pf_value_t_fu[i])
    #         list_X_t = np.array(list_X_t)
    #         list_W_previous = np.array(list_W_previous)
    #         list_pf_value_previous = np.array(list_pf_value_previous)
    #         list_dailyReturn_t = np.array(list_dailyReturn_t)
    #
    #         # for each batch, train the network to maximize the reward
    #         actor.train(list_X_t, list_W_previous,
    #                     list_pf_value_previous, list_dailyReturn_t)
    #     eval_perf(e, data, actor, total_steps_train, total_steps_val, w_init_eval)
