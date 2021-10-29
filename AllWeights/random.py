import torch
import torch.nn as nn
import numpy as np

class Random(object):
    def __init__(self,num_tasks):
        super(Random,self).__init__()
        self.num_tasks = num_tasks
        
        
    def __call__(self, losses):
        #self.all_loss = losses
        sum_loss = 0
        weights = np.random.dirichlet(np.ones(self.num_tasks),size=1)*self.num_tasks
        weights = torch.from_numpy(weights)
        weights = weights.squeeze()
        
        print(weights)
        for i in range(self.num_tasks):
            sum_loss += losses[i]*weights[i]
        
        return sum_loss