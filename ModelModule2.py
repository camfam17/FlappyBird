import torch
from torch import nn
import Main
from collections import OrderedDict



# Model class extends nn.Module
# Takes init arguments like fitness_function and network
class Model(nn.Module):
	
	def __init__(self, network, fitness_function, name='Agent', device='cpu'):
		super().__init__()
		
		self.network = network
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
		# x = x.to(self.device)
		return self.network(x)
	
	def predict(self, points, dead=False, *data):
		
		# print(f'Points: {points}, dead: {dead}, data:{data}')
		
		self.frames += 1
		self.points = points
		self.in_data = data
		
		# Kill the player after 100 points. If they've played a 100 point game then it's likely perfect
		if points > 100:
			dead = True
			# save(self, 'SelfDestruct100points.pth')
		
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

# Selection functions


# Fitness functions


# Models
# This is where the selection of models will go. 
# They will be ordered_dict dictionaries:
# when an exeriment is run and the list of agents is created you will create a new model in that for loop with these dictionaries (deepcopied?) as arguments
# This should create completely separate instances of models (class) and their networks (variable)


# Create an experiment class that takes arguments like model, selection_functions, fitness_functions, generations, population size etc that will run an experiment and save the results.

if __name__ == '__main__':
    
	print('hi')