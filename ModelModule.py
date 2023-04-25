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
		
		self.mutations = 0
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
			save(self, 'SelfDestruct100points.pth')
		
		if dead:
			self.playing = False
			self.calculate_fitness(self, self.in_data)
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
		self.mutations += 1
		for param in self.parameters():
			if random.random() > 0.5:
				param.data += mutation_rate * torch.randn_like(param)
	
	
	def play_game(self):
		
		self.playing = True
		self.controller = Main.Controller(self)
	
	
	def reset(self):
		self.playing = False
		self.frames = 0
		self.points = 0
		self.fitness = 0
	
	def to_string(self, _print=True):
		s = f'{self.name}, Fitness: {self.fitness}, Mutations: {self.mutations}'
		if _print:
			print(s)
		
		return s


class ArgsModel(Model):
	
	def __init__(self, net, fitness_func, **kwargs):
		super().__init__(**kwargs)
		self.calculate_fitness = fitness_func
		
		self.network = net
		pass
	
	#Overridden
	def data_to_tensor(self, *data):
		# Convert data into torch.Tensor. Dependent on each model implementation
		
		t = []
		for d in data:
			t += d
		
		return torch.tensor(t, dtype=torch.float, device=self.device)
		# eg: return torch.tensor(data)

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
		
		net = nn.Sequential(OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('hidden1', nn.Linear(in_features=10, out_features=1)), ('sigmoid', nn.Sigmoid())
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
		# if self.points == 0:
		# 	self.fitness = 0
		# else:
		# 	self.fitness = self.frames/800 + self.points
		self.fitness = self.frames/800 + self.points



def framesover800(self, *in_data):
		self.fitness = self.frames/800 + self.points

def no_points_for_frames(self, *in_data):
	if self.points == 0:
		return 0
	else:
		return self.points

def save(model, name):
    MODEL_PATH = Path('models1')
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

def roulette_wheel_select(agents, n_pointers=None, reduction=0.5):
	print('Roulette')
	
	if n_pointers is None:
		n_pointers = 2
	
	n_agents = len(agents)
	distance_between_pointers = n_agents/n_pointers
	
	selected_agents = []
	while len(selected_agents) < reduction*n_agents:
		R = random.randint(0, int(distance_between_pointers))
		R_sample = []
		for i in range(1, n_pointers):
			print(f'i: {i}, R: {R}, i*R: {i*R}')
			R_sample.append(agents[i*R])
		R_sample.sort(key= lambda x: x.fitness, reverse=True)
		selected_agents.append(copy.deepcopy(R_sample[0]))
	
	return copy.deepcopy(selected_agents)


n1 = nn.Sequential(OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('sigmoid1', nn.Sigmoid()),
				 				 ('hidden1', nn.Linear(in_features=10, out_features=10)), ('sigmoid2', nn.Sigmoid()),
								 ('hidden2', nn.Linear(in_features=10, out_features=1)), ('sigmoid3', nn.Sigmoid())
				   			]))


population_size = 100
generations = 100
mutation_rate = 0.1
if __name__ == '__main__':
	
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	### Create population ###
	agents = []
	for i in range(population_size):
		# agents.append(FCNN4(name='Agent ' + str(i), device=device).to(device))
		# agents.append(FCNN4(name='Agent ' + str(i), device=device).to(device))
		agents.append(ArgsModel(net=copy.deepcopy(n1), fitness_func=no_points_for_frames, name='Agent ' + str(i), device=device))
		agents[-1].mutate(mutation_rate=0.5)
	
	
	for gen in range(generations):
		print(f'Generation: {gen}')
		
		### Make them play ### NOTE parallelize this section
		for i, agent in enumerate(agents):
			agent.play_game()
			while True:
				if not agent.playing:
					print('i', str(i), agent.to_string(_print=False))
					importlib.reload(Main)
					break
		
		### Evaluate fitness ###
		# sort from best fitness (max) to worst
		# agents.sort(key=lambda x: x.fitness, reverse=True)
		
		# agents = proportionate_select(agents)
		### Apply Selection ###
		selected_agents = roulette_wheel_select(agents, 5)
		
		# Mutate & restore population
		mutated_selected_agents = copy.deepcopy(selected_agents)
		for i in range(len(mutated_selected_agents)):
			mutated_selected_agents[i].mutate(mutation_rate)
		
		agents = selected_agents + mutated_selected_agents
		
		### Save best models? ###
		for i in range(5):
			save(agents[i], f'Gen{gen} {agents[i].name} Top{i+1} Fit {int(agents[i].fitness)}.pth')
		
		### Apply selection ### (truncation fitness)
		# agents = agents[ : int(len(agents)/2)] # Kill worst 50% of agents
		# agents = agents + mutate(copy.deepcopy(agents)) # Keep the best 50%


# Selection functions
# Fitness functions