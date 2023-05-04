import torch
from torch import nn
import Main
from collections import OrderedDict
import numpy as np
import copy
import importlib
import random
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model class extends nn.Module
# Takes init arguments like fitness_function and network
class Model(nn.Module):
	
	def __init__(self, network, fitness_function, name='AgentX', device='cpu'):
		super().__init__()
		
		self.network = nn.Sequential(network).to(device)
		self.calc_fitness = fitness_function
		self.name = name
		self.device = device
		
		self.mutations = 0
		self.playing = False
		self.frames = 0
		self.points = 0
		self.fitness = 0
		
		for param in self.parameters():
			param.requires_grad = False
	
	def forward(self, x):
		x = x.to(self.device)
		return self.network(x)
	
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
			self.calc_fitness(self, self.in_data)
			return False
		
		X = self.data_to_tensor(*data)
		
		self.eval()
		output = self.forward(X.to(self.device))
		
		return output > 0.5
	
	def data_to_tensor(self, *data):
		# Convert data into torch.Tensor. Dependent on each model implementation
		
		t = []
		for d in data:
			t += d
		
		return torch.tensor(t, dtype=torch.float, device=self.device)
		# eg: return torch.tensor(data)
	
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
	
	def mutate(self, mutation_magnitude=0.1, mutate_all=False):
		# self.mutations += 1
		for param in self.parameters():
			# if random.random() > 0.5:
			# mutmat = mutation_magnitude * torch.randn_like(param, dtype=param.dtype)
			mutmat = mutation_magnitude * (2*torch.rand_like(param, dtype=param.dtype) - 1)
			
			# randomly zero some of the values i.e. not all weights are mutated.
			if not mutate_all:
				binmat = np.random.randint(2, size=mutmat.size())
				mutmat *= torch.from_numpy(binmat).to(device)
			
			# print('mutatedmat\n', mutmat)
			
			param.data += mutmat


############### Mutation functions ###############



############### Selection functions ###############
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


############### Fitness functions ###############
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

# Models
# This is where the selection of models will go. 
# They will be ordered_dict dictionaries:
# when an exeriment is run and the list of agents is created you will create a new model in that for loop with these dictionaries (deepcopied?) as arguments
# This should create completely separate instances of models (class) and their networks (variable)
n1 = OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('sigmoid1', nn.Sigmoid()),
				   ('hidden1', nn.Linear(in_features=10, out_features=10)), ('sigmoid2', nn.Sigmoid()),
				   ('hidden2', nn.Linear(in_features=10, out_features=1)), ('sigmoid3', nn.Sigmoid())
				   ])

n1tanh = OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('tanh1', nn.Tanh()),
				   ('hidden1', nn.Linear(in_features=10, out_features=10)), ('tanh2', nn.Tanh()),
				   ('hidden2', nn.Linear(in_features=10, out_features=1)), ('sigmoid3', nn.Sigmoid())
				   ])

n1relu = OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('relu1', nn.ReLU()),
				   ('hidden1', nn.Linear(in_features=10, out_features=10)), ('relu2', nn.ReLU()),
				   ('hidden2', nn.Linear(in_features=10, out_features=1)), ('sigmoid3', nn.Sigmoid())
				   ])

n2 = OrderedDict([('input', nn.Linear(in_features=4, out_features=128)), #('relu1', nn.ReLU()),
                  ('linear1', nn.Linear(in_features=128, out_features=64)), ('relu2', nn.ReLU()),
                  ('linear2', nn.Linear(in_features=64, out_features=32)), ('relu3', nn.ReLU()),
                  ('linear3', nn.Linear(in_features=32, out_features=16)), ('relu4', nn.ReLU()),
                  ('output', nn.Linear(in_features=16, out_features=1)), ('sigmoid1', nn.Sigmoid())
                  ])

# Create an experiment class that takes arguments like model, selection_functions, fitness_functions, generations, population size etc that will run an experiment and save the results.

class Experiment():

	def __init__(self, population_size, generations, model, fitness_function, selection_function):
		
		self.population_size = population_size
		self.generations = generations
		self.model = model
		self.fitness_function = fitness_function
		self.selection_function = selection_function
		
		pass
	
	def start(self):
		
		print('Starting Experiment')
		
		### Create population ###
		agents = []
		for i in range(self.population_size):
			agents.append(Model(network=copy.deepcopy(self.model), fitness_function=self.fitness_function, name='Agent ' + str(i), device=device))
			agents[-1].mutate(mutate_all=True)
			pass
		
		
		for gen in range(generations):
			
			print('Generation', gen)
			
			for i, agent in enumerate(agents):
				print(list(agent.parameters()))
				agent.play_game()
				while True:
					if not agent.playing:
						print('i', str(i), agent.to_string(_print=False))
						importlib.reload(Main)
						break
			
			# for agent in agents:
			# 	agent.mutate(mutation_magnitude=1)
			
			### Apply Selection ###
			selected_agents = self.selection_function(agents, 5)
			
			# Mutate & restore population
			mutated_selected_agents = copy.deepcopy(selected_agents)
			for i in range(len(mutated_selected_agents)):
				mutated_selected_agents[i].mutate()
			
			agents = selected_agents + mutated_selected_agents


population_size = 100
generations = 100
if __name__ == '__main__':
    
	print('hi')
	
	exp1 = Experiment(population_size=100, generations=100, model=n1relu, fitness_function=no_points_for_frames, selection_function=roulette_wheel_select)
	exp1.start()
	
	'''
	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
	### Create population ###
	agents = []
	for i in range(population_size):
		# agents.append(FCNN4(name='Agent ' + str(i), device=device).to(device))
		# agents.append(FCNN4(name='Agent ' + str(i), device=device).to(device))
		agents.append(Model(network=copy.deepcopy(n1relu), fitness_function=no_points_for_frames, name='Agent ' + str(i), device=device))
		agents[-1].mutate(mutate_all=True)
	
	
	for gen in range(generations):
		
		print('Generation', gen)
		
		for i, agent in enumerate(agents):
			print(list(agent.parameters()))
			agent.play_game()
			while True:
				if not agent.playing:
					print('i', str(i), agent.to_string(_print=False))
					importlib.reload(Main)
					break
		
		# for agent in agents:
		# 	agent.mutate(mutation_magnitude=1)
		
		### Apply Selection ###
		selected_agents = roulette_wheel_select(agents, 5)
		
		# Mutate & restore population
		mutated_selected_agents = copy.deepcopy(selected_agents)
		for i in range(len(mutated_selected_agents)):
			mutated_selected_agents[i].mutate()
		
		agents = selected_agents + mutated_selected_agents
		
	'''