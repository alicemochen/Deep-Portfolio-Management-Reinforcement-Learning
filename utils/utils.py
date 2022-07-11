import numpy as np
from matplotlib import pyplot as plt

from model.environment import TradeEnv


def get_random_action(m):
    random_vec = np.random.rand(m+1)
    return random_vec/np.sum(random_vec)


def get_max_draw_down(xs):
    xs = np.array(xs)
    i = np.argmax(np.maximum.accumulate(xs) - xs)  # end of the period
    j = np.argmax(xs[:i])  # start of period

    return xs[j] - xs[i]


def eval_perf(e, data, actor, eval_index, eval_steps, w_init_test, config, save_dir=None):
    """
    This function evaluates the performance of the different types of agents.

    """
    list_weight_end_val = list()
    list_pf_end_training = list()
    list_pf_min_training = list()
    list_pf_max_training = list()
    list_pf_mean_training = list()
    list_pf_dd_training = list()

    #######TEST#######
    # environment for trading of the agent
    env_eval = TradeEnv(data, config)

    # initialization of the environment
    # set t to total_steps_train for evaluation
    state_eval, done_eval = env_eval.reset(w_init_test, config["dict_test"]["pf_init_test"], t=eval_index)

    # first element of the weight and portfolio value
    p_list_eval = [config["dict_test"]["pf_init_test"]]
    w_list_eval = [w_init_test]

    for k in range(eval_index, eval_index + eval_steps - int(config["dict_hp_pb"]["length_tensor"] / 2)):
        X_t = state_eval[0].reshape([-1] + list(state_eval[0].shape))
        W_previous = state_eval[1].reshape([-1] + list(state_eval[1].shape))
        pf_value_previous = state_eval[2]
        # compute the action
        action = actor.compute_W(X_t, W_previous)
        # step forward environment
        state_eval, reward_eval, done_eval = env_eval.step(action)

        X_next = state_eval[0]
        W_t_eval = state_eval[1]
        pf_value_t_eval = state_eval[2]

        dailyReturn_t = X_next[-1, :, -1]
        # print('current portfolio value', round(pf_value_previous,0))
        # print('weights', W_previous)
        p_list_eval.append(pf_value_t_eval)
        w_list_eval.append(W_t_eval)

    list_weight_end_val.append(w_list_eval[-1])
    list_pf_end_training.append(p_list_eval[-1])
    list_pf_min_training.append(np.min(p_list_eval))
    list_pf_max_training.append(np.max(p_list_eval))
    list_pf_mean_training.append(np.mean(p_list_eval))

    list_pf_dd_training.append(get_max_draw_down(p_list_eval))

    print('End of test PF value:', round(p_list_eval[-1]))
    print('Min of test PF value:', round(np.min(p_list_eval)))
    print('Max of test PF value:', round(np.max(p_list_eval)))
    print('Mean of test PF value:', round(np.mean(p_list_eval)))
    print('Max Draw Down of test PF value:', round(get_max_draw_down(p_list_eval)))
    print('End of test weights:', w_list_eval[-1])
    plt.title('Portfolio evolution (validation set) episode {}'.format(e))
    plt.plot(p_list_eval, label='Agent Portfolio Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save_dir is not None:
        plt.savefig(f"{save_dir}//evolution_episode_{e}.png", bbox_inches='tight')
    plt.clf()
    plt.title('Portfolio weights (end of validation set) episode {}'.format(e))
    plt.bar(np.arange(actor.nb_stocks + 1), list_weight_end_val[-1])
    plt.xticks(np.arange(actor.nb_stocks + 1), ['Money'] + actor.config['names'], rotation=45)
    if save_dir is not None:
        plt.savefig(f"{save_dir}//end_weights_episode_{e}.png", bbox_inches='tight')
    plt.clf()
    names = ['Money'] + actor.config['names']
    w_list_eval = np.array(w_list_eval)
    for j in range(1, actor.nb_stocks + 1):
        plt.plot(w_list_eval[1:, j], label='Weight Stock {}'.format(names[j]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.5)
    if save_dir is not None:
        plt.savefig(f"{save_dir}//weights_evolution_episode_{e}.png", bbox_inches='tight')
    plt.clf()

