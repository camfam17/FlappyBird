import torch
from torch import nn
from collections import OrderedDict
import Main
import time
import importlib
import statistics as stats
import math
from pathlib import Path
import random

# Fully connected neural network. Can take four/8 inputs: the corners of the pipe in front of it. Much cheaper to compute
class FCNN8(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.playing = False
        self.frames = 0
        self.points = 0
        
        self.network = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=8, out_features=128)), #('relu1', nn.ReLU()),
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


class FCNN4(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.playing = False
        self.frames = 0
        self.points = 0
        self.fitness = 0
        
        self.network = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=4, out_features=128)), #('relu1', nn.ReLU()),
                                                  ('linear1', nn.Linear(in_features=128, out_features=64)), ('relu2', nn.ReLU()),
                                                  ('linear2', nn.Linear(in_features=64, out_features=32)), ('relu3', nn.ReLU()),
                                                  ('linear3', nn.Linear(in_features=32, out_features=16)), ('relu4', nn.ReLU()),
                                                  ('output', nn.Linear(in_features=16, out_features=1)), ('sigmoid1', nn.Sigmoid())
                                                 ]))
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.network(x)
    
    
    def mutate(self, mutation_rate):
        for param in self.parameters():
            param.data += mutation_rate * torch.randn_like(param)
    
    
    def play_game(self):
        
        self.playing = True
        self.controller = Main.Controller(self)
    
    
    def predict(self, bird_coords, pipe_coords, points, dead=False): # returns boolean (True for jump, False for don't jump)
        
        if dead:
            #save state or something
            # self.controller = None
            self.playing = False
            self.calculate_fitness(bird_coords, pipe_coords)
            return False
        
        self.frames += 1
        self.points = points
        
        bird_coords = list(bird_coords)
        # pipe_coords = sum(pipe_coords, [])
        
        X = bird_coords + pipe_coords
        
        # X --> make a tensor of bird_coords & pipe_coords
        # print('inference')
        X = torch.Tensor(X).type(torch.float)
        X = X.to(device)
        # print(X)
        output = self.forward(X)
        # print('output', output)
        
        
        return output > 0.5
    
    def calculate_fitness(self, bird, pipe):
        # fitness function might need to take in difference in y coordinates 
        # between the bird the next pipe i.e. a bird that died closer to the gap in the pipes is more fit than other birds
        
        # may need dto punish the agent for flying up off the screen. Initially they all realise that continuously flapping is better than not flapping at all 
        
        #NB NB NOTE NOTE 
        # Make fiteness function EXCLUSIVELY  the distance (pythagorean/ eucldian distance?) from the mid_point of the pipes
        
        print(bird)
        print(pipe)
        
        # Wait - inverse of the distance?
        # self.fitness = 1/1+(math.sqrt(math.pow( int(bird[0] - pipe[0]) , 2) + math.pow( int(bird[1] - pipe[1]) , 2))) + 100*self.points # this doesnt make sense. youre trying to minimise distance but maximise points
        
        self.fitness = self.frames/800 + self.points
        
        pass
    
    def reset(self):
        self.playing = False
        self.frames = 0
        self.points = 0



# Convolutional neural network based on input image --> will be hugely computationally expensive
class CNN(nn.Module):
    def __init__(self):
        super().__init__()


# Add ability to save a game. Save all the coordinates of the pipes, save the times that the player jumpa etc
# have the ability replay only the pipes or only the bird
# have the ability to replay the game as video or replay the same pattern of pipes

def save(model, name):
    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True, exist_ok='True')
    
    # 2. Create model save path
    MODEL_NAME = name + '.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    # 3. Save them model state_dict()
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

mutation_power = 0.1
population_size = 200
generations = 50
if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create population
    agents = []
    for i in range(population_size):
        agents.append(FCNN4().to(device))
        # agents[-1].mutate()
    
    # agents = []
    # model = FCNN4()
    # model.load_state_dict(torch.load('models/Generation 6 Best Fit 227.55021518418383.pth'))
    # model = model.to(device=device)
    # agents = [model] * 5
    
    for gen in range(generations):
        
        print('Generation', gen)
        
        # Make them play the game
        for i in range(len(agents)):
            print(f'Agent {i}', end='q')
            agents[i].play_game()
            while True:
                if agents[i].fitness > 5:
                        save(agents[i], 'Score' + str(agents[i].fitness) +'model' + str(i) + 'gen' + str(gen) + str(random.randint(0, 100000)))
                if not agents[i].playing:       
                    print('fitness = ', agents[i].fitness)             
                    importlib.reload(Main)
                    break
                time.sleep(5000)
        
        for a in agents:
            if a.frames > 80:
                print(a.frames, end=', ')
        print('')
        
        
        # Evaluate fitness
        # Create a number representing fitness level:
        # individuals with the most points will have highest fitness
        # the number of frames that the agent stayed alive may also count (differentiate between agents with the same number of points)
        
        
        
        # Select best individuals
        # Choose the agents with the highest fitness value. 
        # Randomness will likely be necessary here
        
        # fitnesses = []
        # for agent in agents:
        #     fitnesses.append(agent.fitness)
        
        # median = stats.median(fitnesses)
        # print(f'len agents {len(agents)}')
        # for i, agent in enumerate(agents):
        #     if agent.fitness < median:
        #         agents.pop(i)
        # print(f'len agents {len(agents)}')
        
        print('1', [x.fitness for x in agents])
        agents.sort(key=lambda x: x.fitness, reverse=True)
        print('2', [x.fitness for x in agents])
        
        agents = agents[ : int(len(agents)/2)]
        
        save(agents[0], f'Generation {gen} Best Fit {agents[0].fitness}.pth')
        
        # return population to full size
        for agent in agents:
            agent.reset()
        
        
        agents = agents + agents
        print('\n\nBefore Mutate:', next(agents[0].parameters()))
        for agent in agents[60:]:
            agent.mutate(mutation_rate=0.3)
        print('\n\nAfter Mutate:', next(agents[0].parameters()))
        
        
    print('end')
    
#### Genetic algorithm: ####
# 1) Create population
# 2) Test the population (make predictions)
# 3) Evaluate fitness of individuals
# 4) Select best individuals
# 5) Mutation
# 6) GOTO (2)