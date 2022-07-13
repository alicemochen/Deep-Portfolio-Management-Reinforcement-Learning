import numpy as np


class PVM(object):
    '''
    This is the memory stack called PVM in the paper
    '''

    def __init__(self, config, total_steps, w_init):

        # initialization of the memory
        # we have a total_step_times the initialization portfolio tensor
        self.memory = np.transpose(np.array([w_init] * total_steps))
        self.sample_bias = config["sample_bias"]
        self.total_steps = total_steps
        self.batch_size = config["dict_hp_pb"]["batch_size"]

    def get_W(self, t):
        # return the weight from the PVM at time t
        return self.memory[:, t]

    def update(self, t, w):
        # update the weight at time t
        self.memory[:, t] = w

    def draw(self):
        '''
        returns a valid step so you can get a training batch starting at this step
        '''
        while 1:
            z = np.random.geometric(p=self.sample_bias)
            tb = self.total_steps - self.batch_size + 1 - z
            if tb >= 0:
                return tb

    def test(self):
        # just to test
        return self.memory
