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
	
	def mutate(self, mutation_rate=0.1):
		# self.mutations += 1
		for param in self.parameters():
			# if random.random() > 0.5:
			mutmat = mutation_rate * torch.randn_like(param, dtype=param.dtype)

			# randomly zero some of the values i.e. not all weights are mutated.
			binmat = np.random.randint(2, size=mutmat.size())
			print('binmat\n', binmat)
			mutmat *= binmat
			
			print('mutatedmat\n', mutmat)
			
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
		print(param)
	
	m1.mutate()
	for param in m1.parameters():
		print(param)
	