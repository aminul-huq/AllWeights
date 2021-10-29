import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Uncertainty(object):
    def __init__(self,num_tasks):
        super(Uncertainty,self).__init__()
        self.num_tasks = num_tasks
        params = torch.ones(num_tasks, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        
    def get_params(self):
        return self.params
    
    def __call__(self, losses):
        
        sum_loss = 0
        
        for i in range(len(losses)):
            precision = torch.exp(-self.params[i])
            sum_loss += torch.sum(precision*losses[i] + self.params[i],-1)
        
        return torch.mean(sum_loss)