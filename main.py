import pickle

from utils.data_utils import get_stock_features_default

from model.policy import Policy
import tensorflow.compat.v1 as tf
from model.environment import TradeEnv
import os
import json


def problem_setup(config, input_dir):
    list_stock = config["names"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    feature_func = globals()[config["feature_func"]]
    data = feature_func(list_stock, start_date, end_date)
    config["trading_period"] = data.shape[2]
    config["nb_feature_map"] = data.shape[0]
    config["nb_stocks"] = data.shape[1]
    config["input_dir"] = input_dir
    return config, data


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.curdir)
    input_dirs = [f"{ROOT_DIR}\\run_directory\\sector_rotation", f"{ROOT_DIR}\\run_directory\\tech_stocks"]
    tf.reset_default_graph()
    tf.disable_eager_execution()
    sess = tf.Session()
    for input_dir in input_dirs:
        config = json.load(open(f"{input_dir}\\config.json"))
        config, data = problem_setup(config, input_dir)
        env = TradeEnv(data, config)
        actor = Policy(sess, env, config)  # policy initialization
        sess.run(tf.global_variables_initializer())
        actor.train()
        # pickle.dump(actor,  open(f"{input_dir}\\policy_actor.pkl", 'wb'))