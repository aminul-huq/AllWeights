import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DWA(object):
    def __init__(self,num_tasks,total_epochs):
        super(DWA,self).__init__()
        self.num_tasks = num_tasks
        self.total_epochs = total_epochs
        self.loss_matrix = torch.ones(num_tasks, total_epochs)
        
    
    def __call__(self, losses,epoch):
        sum_loss = 0
        total = 0
        t = 2.0
        temp = torch.zeros(self.num_tasks)
        
        if epoch == 0 or epoch == 1 :
            for i in range(self.num_tasks):
                self.loss_matrix[i,epoch] = losses[i]
                sum_loss += losses[i]
            return sum_loss
        else:
            
            for i in range(self.num_tasks):
                temp[i] = self.loss_matrix[i,epoch-1]/self.loss_matrix[i,epoch-2]
                total += torch.exp(temp[i]/t)
            for i in range(self.num_tasks):
                self.loss_matrix[i,epoch] = losses[i]
                sum_loss += self.num_tasks * losses[i] * (torch.exp(temp[i]/t)/total)
        
        return sum_loss