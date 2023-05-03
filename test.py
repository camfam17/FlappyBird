import torch
from torch import nn
from collections import OrderedDict
import random
import numpy as np


class model(nn.Module):
    
	def __init__(self, net):
		super().__init__()
		
		self.network = net
		
		for param in self.parameters():
			param.requires_grad = False
	
	# def mutate(self, mutation_rate=0.1):
	# 	# self.mutations += 1
	# 	for param in self.parameters():
	# 		# if random.random() > 0.5:
	# 		mutmat = mutation_rate * torch.randn_like(param, dtype=param.dtype)

	# 		# randomly zero some of the values i.e. not all weights are mutated.
	# 		binmat = np.random.randint(2, size=mutmat.size())
	# 		print('binmat\n', binmat)
	# 		mutmat *= binmat
			
	# 		print('mutatedmat\n', mutmat)
			
	# 		param.data += mutmat
	
	def mutate(self, mutation_magnitude=0.1, mutate_all=False):
		# self.mutations += 1
		for param in self.parameters():
			# if random.random() > 0.5:
			# mutmat = mutation_magnitude * torch.randn_like(param, dtype=param.dtype)
			mutmat = mutation_magnitude * (2*torch.rand_like(param, dtype=param.dtype) - 1)
			
			# randomly zero some of the values i.e. not all weights are mutated.
			if not mutate_all:
				binmat = np.random.randint(2, size=mutmat.size())
				print('binmat\n', binmat)
				mutmat *= torch.from_numpy(binmat)
			
			# print('mutatedmat\n', mutmat)
			
			param.data += mutmat


if __name__ == '__main__':
	
	print('hi')
	
	net1 = nn.Sequential(OrderedDict([ ('input', nn.Linear(in_features=4, out_features=10)), ('sigmoid1', nn.Sigmoid()),
				 				 #('hidden1', nn.Linear(in_features=10, out_features=10)), ('sigmoid2', nn.Sigmoid()),
								 ('hidden2', nn.Linear(in_features=10, out_features=1)), ('sigmoid3', nn.Sigmoid())
				   			]))
	
	m1 = model(net1)
	print(m1)
	
	for param in m1.parameters():
		print('Before mutation:\n', param)
	
	m1.mutate(mutate_all=True)
	
	for param in m1.parameters():
		print('After mutation:\n', param)
	
	
	
	# t1 = torch.rand(size=(1000, ))
	
	# # t1 = torch.from_numpy(np.random.randint(2, size=(1000, )))
	# print(t1)
	
	# print(max(t1))
	# print(min(t1))
	
	# t2 = torch.randn_like(t1, dtype=t1.dtype)
	# print(t2)
	
	# print(max(t2))
	# print(min(t2))
	
	
	# n1 = nn.Sequential(OrderedDict([('input', nn.Linear(in_features=3, out_features=128)), #('relu1', nn.ReLU()),
    #               ('linear1', nn.Linear(in_features=128, out_features=64)), ('relu2', nn.ReLU()),
    #               ('linear2', nn.Linear(in_features=64, out_features=32)), ('relu3', nn.ReLU()),
    #               ('linear3', nn.Linear(in_features=32, out_features=16)), ('relu4', nn.ReLU()),
    #               ('output', nn.Linear(in_features=16, out_features=1)), ('sigmoid1', nn.Sigmoid())
    #               ]))
	
	# n11 = list(n1.parameters())[2]
	
	# print(n11)
	# print(torch.max(n11))
	# print(torch.min(n11))
	
	# x = torch.randint(-1,2, size=(1000, ))
	# x = 2*torch.rand((1000, )) - 1
	# # x = torch.rand((1000, ))
	# # x = torch.randn((1000, ))
	# # x = torch.tanh(x)
	# print(x)
	# print(torch.max(x))
	# print(torch.min(x))