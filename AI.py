import torch
import torch.nn
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super().__init__()


#### Genetic algorithm: ####
# 1) Create population
# 2) Test the population (make predictions)
# 3) Evaluate fitness of individuals
# 4) Select best individuals
# 5) Mutation
# 6) GOTO (2)