# Optimizer

import math
import torch
import torch.optim as optim

# Implementing the Adam optimizer with shared states

class SharedAdam(optim.Adam): # object that inherits from optim.Adam

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay) # inheriting from the tools of optim.Adam
        for group in self.param_groups: # self.param_groups contains all the attributes of the optimizer, including the parameters to optimize (the weights of the network) contained in self.param_groups['params']
            for p in group['params']: # for each tensor p of weights to optimize
                state = self.state[p] # at the beginning, self.state is an empty dictionary so state = {} and self.state = {p:{}} = {p: state}
                state['step'] = torch.zeros(1) # counting the steps: state = {'step' : tensor([0])}
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_() # the update of the adam optimizer is based on an exponential moving average of the gradient (moment 1)
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_() # the update of the adam optimizer is also based on an exponential moving average of the squared of the gradient (moment 2)

    # Sharing the memory
    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()
                state['exp_avg'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()
                state['exp_avg_sq'].share_memory_() # tensor.share_memory_() acts a little bit like tensor.cuda()

    # Performing a single optimization step of the Adam algorithm (see algorithm 1 in https://arxiv.org/pdf/1412.6980.pdf)
    def step(self):
        super(SharedAdam,self).step()
