import torch
from torch import nn
import Main
from collections import OrderedDict
from abc import abstractmethod
import importlib
from pathlib import Path
import copy
import random

class Model(nn.Module):
	
	def __init__(self, name='AgentX', device='cpu'):
		super().__init__()
		self.name = name
		self.device = device
		
		self.playing = False
		self.frames = 0
		self.points = 0
		self.fitness = 0
		
		self.network = self.create_network()
		
		for param in self.parameters():
			param.requires_grad = False
	
	
	@abstractmethod
	def create_network(self) -> nn.Module:
		pass
	
	
	def forward(self, x):
		# x = x.to(self.device)
		return self.network(x)
	
	
	@abstractmethod
	def data_to_tensor(self, *data):
		# Convert data into torch.Tensor. Dependent on each model implementation
		pass
	
	
	def predict(self, points, dead=False, *data):
		
		# print(f'Points: {points}, dead: {dead}, data:{data}')
		
		self.frames += 1
		self.points = points
		self.in_data = data
		
		# Kill the player after 100 points. If they've played a 100 point game then it's likely perfect
		if points > 100:
			dead = True
		
		if dead:
			self.playing = False
			self.calculate_fitness(self.in_data)
			return False
		
		X = self.data_to_tensor(*data)
		
		self.eval()
		output = self.forward(X.to(self.device))
		
		return output > 0.5
	
	
	@abstractmethod
	def calculate_fitness(self, *in_data):
		# Create fitness function. Dependent on each model implementation
		# The *in_data argument is the data fed to the NN at that instant
		pass
	
	
	def mutate(self, mutation_rate=0.1):
		for param in self.parameters():
			param.data += mutation_rate * torch.randn_like(param)
	
	
	def play_game(self):
		
		self.playing = True
		self.controller = Main.Controller(self)
	
	
	def reset(self):
		self.playing = False
		self.frames = 0
		self.points = 0
		self.fitness = 0


class FCNN4(Model):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	
	#Overridden
	def create_network(self) -> nn.Module:
		
		net = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=4, out_features=128)), #('relu1', nn.ReLU()),
                                                  ('linear1', nn.Linear(in_features=128, out_features=64)), ('relu2', nn.ReLU()),
                                                  ('linear2', nn.Linear(in_features=64, out_features=32)), ('relu3', nn.ReLU()),
                                                  ('linear3', nn.Linear(in_features=32, out_features=16)), ('relu4', nn.ReLU()),
                                                  ('output', nn.Linear(in_features=16, out_features=1)), ('sigmoid1', nn.Sigmoid())
                                                 ]))
		
		return net
	
	
	#Overridden
	def data_to_tensor(self, *data):
		# Convert data into torch.Tensor. Dependent on each model implementation
		
		t = []
		for d in data:
			t += d
		
		return torch.tensor(t, dtype=torch.float, device=self.device)
		# eg: return torch.tensor(data)
	
	
	#Overriden
	def calculate_fitness(self, *in_data):
		self.fitness = self.frames/800 + self.points


class NoHiddenLayers(Model):
	
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
	
	#Overriden
	def create_network(self) -> nn.Module:
		
		net = nn.Sequential(OrderedDict([ ('input', nn.Linear(in_features=4, out_features=1)), ('sigmoid', nn.Sigmoid())
				   			]))
		
		return net
	
	#Overriden
	def data_to_tensor(self, *data):
		
		t = []
		for d in data:
			t += d
		
		return torch.tensor(t, dtype=torch.float, device=self.device)
	
	#Overriden
	def calculate_fitness(self, *in_data):
		self.fitness = self.frames/800 + self.points


def save(model, name):
    MODEL_PATH = Path('models')
    MODEL_PATH.mkdir(parents=True, exist_ok='True')
    
    # 2. Create model save path
    MODEL_NAME = name + '.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    
    # 3. Save them model state_dict()
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def mutate(agents, mutation_rate=0.1):
	for agent in agents:
		agent.mutate(mutation_rate=mutation_rate)
	return agents


def proportionate_select(agents):
	
	# sort agents based on fitness
	agents.sort(key=lambda x: x.fitness, reverse=True)
	# create a separate list of fitnesses
	fitnesses = []
	for agent in agents:
		fitnesses.append(agent.fitness)
	
	# softmax that list
	sfmx = torch.softmax(torch.tensor(fitnesses), dim=0, dtype=torch.float)
	print(sfmx)
	
	new_agents = []
	while len(new_agents) < len(agents):
		R = random.random()
		
		for i, fit in enumerate(fitnesses):
			if fit > R:
				new_agents.append(agents[i])
				break
	
	return new_agents


def roulette_wheel_select(agents, n_pointers=2):
	
	
	
	pass


population_size = 10
generations = 100
mutation_rate = 0.1
if __name__ == '__main__':
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	### Create population ###
	agents = []
	for i in range(population_size):
		agents.append(FCNN4(name='Agent ' + str(i), device=device).to(device))
	
	
	for gen in range(generations):
		print(f'Generation: {gen}')
		
		### Make them play ### NOTE parallelize this section
		for agent in agents:
			agent.play_game()
			while True:
				if not agent.playing:
					print(agent.name + ' fitness =', agent.fitness)
					importlib.reload(Main)
					break
		
		### Evaluate fitness ###
		# sort from best fitness (max) to worst
		# agents.sort(key=lambda x: x.fitness, reverse=True)
		
		agents = proportionate_select(agents)
		
		### Save best models? ###
		for i in range(5):
			save(agents[i], f'Gen{gen} {agents[i].name} Top{i+1} Fit {int(agents[i].fitness)}.pth')
		
		### Apply selection ###
		# agents = agents[ : int(len(agents)/2)] # Kill worst 50% of agents
		# agents = agents + mutate(copy.deepcopy(agents)) # Keep the best 50%
		
		### Mutation ###
		
		for agent in agents:
			agent.reset()
			print(agent.fitness)