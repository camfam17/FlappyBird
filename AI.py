import torch
from torch import nn
import numpy as np
from collections import OrderedDict

# Fully connected neural network. Can take four/8 inputs: the corners of the pipe in front of it. Much cheaper to compute
class FCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.frames = 0
        
        self.network = nn.Sequential(OrderedDict([('linear1', nn.Linear(in_features=8, out_features=32)), ('relu1', nn.ReLU()),
                                                  ('linear2', nn.Linear(in_features=32, out_features=16)), ('relu2', nn.ReLU()),
                                                  ('linear3', nn.Linear(in_features=16, out_features=2)),
                                                 ]))
        
        for param in self.parameters():
            param.requires_grad = False
            print('param', param.data)
        
        # print(list(self.parameters()))
    
    def forward(self, x):
        return self.network(x)
    
    
    def mutate(self):
        
        # Mutate the parameters
        # for key in self.
        
        
        for param in self.parameters():
            param.data += mutation_power * torch.randn_like(param)
        
        pass

# Convolutional neural network based on input image --> will be hugely computationally expensive
class CNN(nn.Module):
    def __init__(self):
        super().__init__()


mutation_power = 0.2
if __name__ == '__main__':
    print('start')
    agent1 = FCNN()
    print('end\n\n\n')
    
    print(list(agent1.parameters())[1].data)
    agent1.mutate()
    print(list(agent1.parameters())[1].data)

#### Genetic algorithm: ####
# 1) Create population
# 2) Test the population (make predictions)
# 3) Evaluate fitness of individuals
# 4) Select best individuals
# 5) Mutation
# 6) GOTO (2)