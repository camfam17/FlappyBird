import torch
from torch import nn
from collections import OrderedDict
import Main
import time
import importlib

# Fully connected neural network. Can take four/8 inputs: the corners of the pipe in front of it. Much cheaper to compute
class FCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.playing = False
        self.frames = 0
        self.points = 0
        
        self.network = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=8, out_features=128)), ('relu1', nn.ReLU()),
                                                  ('linear1', nn.Linear(in_features=128, out_features=64)), ('relu2', nn.ReLU()),
                                                  ('linear2', nn.Linear(in_features=64, out_features=32)), ('relu3', nn.ReLU()),
                                                  ('linear3', nn.Linear(in_features=32, out_features=16)), ('relu4', nn.ReLU()),
                                                  ('output', nn.Linear(in_features=16, out_features=1)), ('sigmoid1', nn.Sigmoid())
                                                 ]))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.network(x)
    
    
    def mutate(self):
        for param in self.parameters():
            param.data += mutation_power * torch.randn_like(param)
    
    
    def play_game(self):
        
        self.playing = True
        self.controller = Main.Controller(self)
    
    
    def predict(self, bird_coords, pipe_coords, points, dead=False): # returns boolean (True for jump, False for don't jump)
        
        # NB important to get the coords of the bird as well (probably)
        
        if dead:
            #save state or something
            # self.controller = None
            self.playing = False
            return False
        
        self.frames += 1
        self.points = points
        
        bird_coords = list(*bird_coords)
        pipe_coords = sum(pipe_coords, [])
        
        X = bird_coords + pipe_coords
        
        # X --> make a tensor of bird_coords & pipe_coords
        print('inference')
        X = torch.Tensor(X).type(torch.float)
        # print(X)
        output = self.forward(X)
        print('output', output)
        
        
        return output > 0.5
    

# Convolutional neural network based on input image --> will be hugely computationally expensive
class CNN(nn.Module):
    def __init__(self):
        super().__init__()


mutation_power = 0.2
nagents = 100
if __name__ == '__main__':
    
    # Create population
    agents = []
    for i in range(nagents):
        agents.append(FCNN())
        agents[-1].mutate()
    
    
    # Make them play the game
    for i in range(len(agents)):
        agents[i].play_game()
        while True:
            if not agents[i].playing:
                importlib.reload(Main)
                break
            time.sleep(5000)
    
    for a in agents:
        print(a.frames, end=', ')
    
    
    # Evaluate fitness
    # Create a number representing fitness level:
    # individuals with the most points will have highest fitness
    # the number of frames that the agent stayed alive may also count (differentiate between agents with the same number of points)
    
    
    
    # Select best individuals
    
    print('end')
    
#### Genetic algorithm: ####
# 1) Create population
# 2) Test the population (make predictions)
# 3) Evaluate fitness of individuals
# 4) Select best individuals
# 5) Mutation
# 6) GOTO (2)