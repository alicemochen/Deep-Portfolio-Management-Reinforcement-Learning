import numpy as np
import tensorflow.compat.v1 as tf

# dataset

#can be changed following the type of stocks studied

from_path = False
path_data = './np_data/inputCrypto.npy'

namesBio=['JNJ','PFE','AMGN','MDT','CELG','LLY']
namesUtilities=['XOM','CVX','MRK','SLB','MMM']
namesTech=['FB','AMZN','MSFT','AAPL','T','VZ','CMCSA','IBM','CRM','INTC']
namesCrypto = ['ETCBTC', 'ETHBTC', 'DOGEBTC', 'ETHUSDT', 'BTCUSDT', 'XRPBTC', 'DASHBTC', 'XMRBTC', 'LTCBTC', 'ETCETH']

# list_stock = namesCrypto

list_stock = ["XLE", "XLP", "XLU", "XLV", "XLB", "XLRE", "XLI", "XLF"]

start_date = "2008-01-01"
end_date = "2022-07-01"


# if data_type == 'Utilities':
#     list_stock = namesUtilities
# elif data_type == 'Bio':
#     list_stock = namesBio
# elif data_type == 'Tech':
#     list_stock = namesTech
# elif data_type == 'Crypto':
#     list_stock = namesCrypto
# else:
#     list_stock = [i for i in range(m)]


###############################dictionaries of the problem###########################
dict_hp_net = {'n_filter_1': 2, 'n_filter_2': 20, 'kernel1_size':(1, 3)}
dict_hp_pb = {'batch_size': 50, 'ratio_train': 0.6,'ratio_val': 0.2, 'length_tensor': 10,
              'ratio_greedy':0.8, 'ratio_regul': 0.1}
dict_hp_opt = {'regularization': 1e-8, 'learning': 9e-2}
dict_fin = {'trading_cost': 0.25/100, 'interest_rate': 0.02/250, 'cash_bias_init': 0.7}
dict_train = {'pf_init_train': 10000, 'w_init_train': 'd', 'n_episodes':2, 'n_batches':10}
dict_test = {'pf_init_test': 10000, 'w_init_test': 'd'}


###############################HP of the network ###########################
n_filter_1 = dict_hp_net['n_filter_1']
n_filter_2 = dict_hp_net['n_filter_2']
kernel1_size = dict_hp_net['kernel1_size']

###############################HP of the problem###########################

# Size of mini-batch during training
batch_size = dict_hp_pb['batch_size']

# Number of the columns (number of the trading periods) in each input price matrix
n = dict_hp_pb['length_tensor']

ratio_greedy = dict_hp_pb['ratio_greedy']

ratio_regul = dict_hp_pb['ratio_regul']

##############################HP of the optimization###########################


# The L2 regularization coefficient applied to network training
regularization = dict_hp_opt['regularization']
# Parameter alpha (i.e. the step size) of the Adam optimization
learning = dict_hp_opt['learning']

optimizer = tf.train.AdamOptimizer(learning)


##############################Finance parameters###########################

trading_cost= dict_fin['trading_cost']
interest_rate= dict_fin['interest_rate']
cash_bias_init = dict_fin['cash_bias_init']

############################## PVM Parameters ###########################
sample_bias = 5e-5  # Beta in the geometric distribution for online training sample batches


############################## Training Parameters ###########################



pf_init_train = dict_train['pf_init_train']

n_episodes = dict_train['n_episodes']
n_batches = dict_train['n_batches']

############################## Test Parameters ###########################

pf_init_test = dict_test['pf_init_test']


############################## other environment Parameters ###########################


