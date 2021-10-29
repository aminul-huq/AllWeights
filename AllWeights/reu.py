import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ReU(object):
    def __init__(self,num_tasks):
        super(ReU,self).__init__()
        self.num_tasks = num_tasks
        params = torch.ones(num_tasks, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        
    def get_params(self):
        return self.params
    
    def __call__(self, losses):
        
        sum_loss = 0
        for i, loss in enumerate(losses):
            sum_loss += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        
        return sum_loss