import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
import torch.nn.init as init
from operator import itemgetter
from heapq import nsmallest
import time


from flops import *

class ModifiedSqueezeNetModel(torch.nn.Module): # This class create an object 
	def __init__(self):
		super(ModifiedSqueezeNetModel, self).__init__()
		model = models.squeezenet1_1(pretrained=True)
		self.features = model.features
		for param in self.features.parameters():
			param.requires_grad = False

		final_conv = nn.Conv2d(512,2,kernel_size=1)
		self.classifier = nn.Sequential( 
			nn.Dropout(p=0.5),
			final_conv,
			nn.ReLU(inplace=True),
			nn.AvgPool2d(13,stride=1))


	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)
		return x.view(x.size(0),-1)


class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self): # This just reset the filter_rank dictionary of the object
		# self.activations = []
		# self.gradients = []
		# self.grad_index = 0
		# self.activation_to_layer = {}
		self.filter_ranks = {}

	def forward(self, x): # ????
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0 # What is activation index??
		list_output=[]
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				x = module(x)
				x.register_hook(self.compute_rank)
				self.activations.append(x)
				self.activation_to_layer[activation_index] = layer
				activation_index += 1
			elif isinstance(module, torchvision.models.squeezenet.Fire):
				x=module(x)
				x.register_hook(self.compute_rank)
				self.activations.append(x)
				self.activation_to_layer[activation_index] = layer
				activation_index += 1
			else:
				x= module(x)


		x = self.model.classifier(x)
	
		return (x.view(x.size(0),-1))

	def compute_rank(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = \
			torch.sum((activation * grad), dim = 0, keepdim=True).\
				sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data
		
		values = \
			values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
		
		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))
		print("512 flters to prune: ", filters_to_prune)
		return filters_to_prune				

class PrunningFineTuner_Squeezenet:
	def __init__(self, train_path, test_path, model):
		self.train_data_loader = dataset.loader(train_path)
		self.test_data_loader = dataset.test_loader(test_path)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		self.model.train()


	def test(self):
		self.model.eval()
		correct = 0
		total = 0

		for i, (batch, label) in enumerate(self.test_data_loader):
			batch = batch.cuda()
			output = model(Variable(batch))
			pred = output.data.max(1)[1]
			correct += pred.cpu().eq(label).sum()
			total += label.size(0)
		
		print( "Accuracy :", float(correct) / total)
		
		self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.classifier.parameters(), 
					lr=0.01, momentum=0.90)

		for i in range(epoches):
			print ("Epoch: ", i)
			self.train_epoch(optimizer)
			self.test()
		print ("Finished fine tuning.")
		

	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)


		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward() # This is I think the training part.... not sure :(
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()
		self.train_epoch(rank_filters = True)
		self.prunner.normalize_ranks_per_layer()
		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0

		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d ):
				filters = filters + module.out_channels
			if isinstance(module, torchvision.models.squeezenet.Fire):
				filters += module.expand1x1.out_channels + module.expand3x3.out_channels 

		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()

		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 128
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

		iterations = int(iterations * 2.0 / 3)
		print ("Number of prunning iterations to reduce 50% filters", iterations)
		for _ in range(iterations):
			print ("Ranking filters.. ")
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration) # Get filters to prune...
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			print ("Layers that will be prunned", layers_prunned)
			print ("Prunning filters.. ")
			model = self.model.cpu()
			model_1 = model
			for layer_index, filter_index in prune_targets:
				model = prune_squeezenet_conv_layer(model, layer_index, filter_index)

			self.model = model.cuda()
			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print ("Filters prunned", str(message))
			print("This is the accuracy after making chnages...")
			self.test()
			print ("Fine tuning to recover from prunning iteration.")
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9) # ......
			self.train(optimizer, epoches = 10)

		print ("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, epoches = 15)
		torch.save(model, "model_prunned")


def test_single(model,test_single_path):
	test_single_loader = dataset.test_loader(test_single_path)
	model.eval()
	time_ = 0 
	for i, (batch, label) in enumerate(test_single_loader):
		batch = batch.cuda()
		t0 = time.time()
		output = model(Variable(batch))
		t1 = time.time()			
		time_ += t1-t0
		pred = output.data.max(1)[1]
	print ("Avg time taken per image is(over 800 Images..)  ", time_/800)
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type = str, default = "model_prunned")
	parser.add_argument("--train", dest="train", action="store_true")
	parser.add_argument("--prune", dest="prune", action="store_true")
	parser.add_argument("--run", dest="run",action="store_true")
	parser.add_argument("--flops", dest="flops",action="store_true")
	parser.add_argument("--train_path", type = str, default = "train")
	parser.add_argument("--test_path", type = str, default = "test")
	parser.add_argument("--test_single_path", type = str, default = "test_single")
	parser.set_defaults(train=False)
	parser.set_defaults(prune=False)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()

	if args.flops:
		# main_(torch.load("model").cuda())
		main_(torch.load("model_prunned").cuda())
	elif args.run:
		model = torch.load(args.model).cuda()
		test_single(model, args.test_path)
	else:	
		if args.train:
			model = ModifiedSqueezeNetModel().cuda() # Create a model
		elif args.prune:
			model = torch.load("model").cuda()

		fine_tuner = PrunningFineTuner_Squeezenet(args.train_path, args.test_path, model)

		if args.train:
			fine_tuner.train(epoches = 20) # 
			torch.save(model, "model")

		elif args.prune:
			fine_tuner.prune()
