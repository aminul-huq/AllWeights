import torch
import torch.nn as nn

class Equal(object):
    def __init__(self,num_tasks):
        super(Equal,self).__init__()
        self.num_tasks = num_tasks
        
        
    def __call__(self, losses, weight=None):
        #self.all_loss = losses
        sum_loss = 0
        if weight == None:
            weight = 1
        
        for i in range(self.num_tasks):
            sum_loss += losses[i]*weight
        

        return sum_loss