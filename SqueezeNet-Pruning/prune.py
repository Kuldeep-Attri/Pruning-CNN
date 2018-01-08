import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_vgg16_conv_layer(model, layer_index, filter_index):
	_, conv = model.features._modules.items()[layer_index]
	# print("conv is before next_conv:.....", conv)
	next_conv = None
	offset = 1
	while layer_index + offset <  len(model.features._modules.items()):
		res =  model.features._modules.items()[layer_index+offset]
		# print res ," res is: .................."
		if isinstance(res[1], torchvision.models.squeezenet.Fire):#torch.nn.modules.conv.Conv2d):
			next_name, next_conv = res
			break
		offset = offset + 1
	if isinstance(conv, torch.nn.modules.conv.Conv2d):
		print("this is conv2d::")
		new_conv = \
			torch.nn.Conv2d(in_channels = conv.in_channels, \
				out_channels = conv.out_channels - 1,
				kernel_size = conv.kernel_size, \
				stride = conv.stride,
				padding = conv.padding,
				dilation = conv.dilation,
				groups = conv.groups,
				)#bias = conv.bias)
		old_weights = conv.weight.data.cpu().numpy()
		new_weights = new_conv.weight.data.cpu().numpy()

	# 	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	# 	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	# 	new_conv.weight.data = torch.from_numpy(new_weights).cuda()
	# 	bias_numpy = conv.bias.data.cpu().numpy()
	# 	bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	# 	bias[:filter_index] = bias_numpy[:filter_index]
	# 	bias[filter_index : ] = bias_numpy[filter_index + 1 :]
	# 	new_conv.bias.data = torch.from_numpy(bias).cuda()

	if isinstance(conv, torchvision.models.squeezenet.Fire):
		if(filter_index<conv.expand1x1.out_channels):
			new_conv = torchvision.models.squeezenet.Fire(inplanes=conv.inplanes,
				squeeze_planes=conv.squeeze.out_channels,
				expand1x1_planes=conv.expand1x1.out_channels-1,
				expand3x3_planes=conv.expand3x3.out_channels)

			# filter_index =filter_index-conv.expand1x1.out_channels
			old_weights = conv.expand1x1.weight.data.cpu().numpy()
			new_weights = new_conv.expand1x1.weight.data.cpu().numpy()
			new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
			new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
			new_conv.expand1x1.weight.data = torch.from_numpy(new_weights).cuda()
			bias_numpy = conv.expand1x1.bias.data.cpu().numpy()
			bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
			bias[:filter_index] = bias_numpy[:filter_index]
			bias[filter_index : ] = bias_numpy[filter_index + 1 :]
			new_conv.expand1x1.bias.data = torch.from_numpy(bias).cuda()

			new_conv.expand3x3.weight.data=conv.expand3x3.weight.data
			new_conv.expand3x3.bias.data=conv.expand3x3.bias.data
			# old_weights_n = conv.expand3x3.weight.data.cpu().numpy()
			# new_weights_n = new_conv.expand3x3.weight.data.cpu().numpy()
			# new_weights_n[: , :, :, :] = old_weights_n[:, :, :, :]
			# new_conv.expand3x3.weight.data = torch.from_numpy(new_weights_n).cuda()
			# bias_numpy_n = conv.expand3x3.bias.data.cpu().numpy()
			# bias_n = np.zeros(shape = (bias_numpy_n.shape[0]), dtype = np.float32)
			# bias_n[:] = bias_numpy_n[:]
			# new_conv.expand3x3.bias.data = torch.from_numpy(bias_n).cuda()


		if(filter_index>=conv.expand1x1.out_channels):
			new_conv = torchvision.models.squeezenet.Fire(inplanes=conv.inplanes,
				squeeze_planes=conv.squeeze.out_channels,
				expand1x1_planes=conv.expand1x1.out_channels,
				expand3x3_planes=conv.expand3x3.out_channels-1)

			filter_index=filter_index-conv.expand1x1.out_channels
			# ******** This is to handle expand3x3 conv2d *******************************
			old_weights = conv.expand3x3.weight.data.cpu().numpy()
			new_weights = new_conv.expand3x3.weight.data.cpu().numpy()
			new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
			new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
			new_conv.expand3x3.weight.data = torch.from_numpy(new_weights).cuda()
			bias_numpy = conv.expand3x3.bias.data.cpu().numpy()
			bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
			bias[:filter_index] = bias_numpy[:filter_index]
			bias[filter_index : ] = bias_numpy[filter_index + 1 :]
			new_conv.expand3x3.bias.data = torch.from_numpy(bias).cuda()


			new_conv.expand1x1.weight.data=conv.expand1x1.weight.data
			new_conv.expand1x1.bias.data=conv.expand1x1.bias.data
			# old_weights_n = conv.expand1x1.weight.data.cpu().numpy()
			# new_weights_n = new_conv.expand1x1.weight.data.cpu().numpy()
			# new_weights_n[: , :, :, :] = old_weights_n[:, :, :, :]
			# new_conv.expand1x1.weight.data = torch.from_numpy(new_weights_n).cuda()
			# bias_numpy_n = conv.expand1x1.bias.data.cpu().numpy()
			# bias_n = np.zeros(shape = (bias_numpy_n.shape[0]), dtype = np.float32)
			# bias_n[:] = bias_numpy_n[:]
			# new_conv.expand1x1.bias.data = torch.from_numpy(bias_n).cuda()
	

	# print "i am motherfucking starboy:: ",type(next_conv)
	if isinstance(next_conv, torchvision.models.squeezenet.Fire) and not next_conv is None:
		# print("also next conv::")
		next_new_conv = torchvision.models.squeezenet.Fire(inplanes=next_conv.inplanes-1,
				squeeze_planes=next_conv.squeeze.out_channels,
				expand1x1_planes=next_conv.expand1x1.out_channels,
				expand3x3_planes=next_conv.expand3x3.out_channels)


		# print("sg: ", next_new_conv.squeeze.weight.data)
		old_weights = next_conv.squeeze.weight.data.cpu().numpy()
		new_weights = next_new_conv.squeeze.weight.data.cpu().numpy()

		new_weights[:, : filter_index, :, :] = old_weights[: ,: filter_index, :, :]
		new_weights[:, filter_index :, :, :] = old_weights[:, filter_index+1 :, :, :]
		next_new_conv.squeeze.weight.data = torch.from_numpy(new_weights).cuda()

		next_new_conv.expand1x1.weight.data=next_conv.expand1x1.weight.data
		# old_weights_1x1 = next_conv.expand1x1.weight.data.cpu().numpy()
		# new_weights_1x1 = next_new_conv.expand1x1.weight.data.cpu().numpy()
		# new_weights_1x1[:, :, :, :] = old_weights_1x1[:, :, :, :]
		# next_new_conv.expand1x1.weight.data = torch.from_numpy(new_weights_1x1).cuda()

		next_new_conv.expand3x3.weight.data=next_conv.expand3x3.weight.data
		# old_weights_3x3 = next_conv.expand3x3.weight.data.cpu().numpy()
		# new_weights_3x3 = next_new_conv.expand3x3.weight.data.cpu().numpy()
		# new_weights_3x3[:, :, :, :] = old_weights_3x3[:, :, :, :]
		# next_new_conv.expand3x3.weight.data = torch.from_numpy(new_weights_3x3).cuda()

		# next_new_conv.bias.data = next_conv.bias.data
		next_new_conv.squeeze.bias.data = next_conv.squeeze.bias.data
		next_new_conv.expand1x1.bias.data = next_conv.expand1x1.bias.data
		next_new_conv.expand3x3.bias.data = next_conv.expand3x3.bias.data

	if not next_conv is None:
	 	features = torch.nn.Sequential(
	            *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
	            	[new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
	 	del model.features
	 	del conv

	 	model.features = features

	# else:
	# 	#Prunning the last conv layer. This affects the first linear layer of the classifier.
	#  	model.features = torch.nn.Sequential(
	#             *(replace_layers(model.features, i, [layer_index], \
	#             	[new_conv]) for i, _ in enumerate(model.features)))
	#  	layer_index = 0
	#  	old_linear_layer = None
	#  	for _, module in model.classifier._modules.items():
	#  		if isinstance(module, torch.nn.Linear):
	#  			old_linear_layer = module
	#  			break
	#  		layer_index = layer_index  + 1

	#  # 	# if old_linear_layer is None:
	#  # 		# raise BaseException("No linear layer found in classifier")
	# 	# params_per_input_channel = old_linear_layer.in_features / conv.out_channels

	#  # 	new_linear_layer = \
	#  # 		torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
	#  # 			old_linear_layer.out_features)
	 	
	#  	# old_weights = old_linear_layer.weight.data.cpu().numpy()
	#  	# new_weights = new_linear_layer.weight.data.cpu().numpy()	 	

	#  	new_weights[:, : filter_index * params_per_input_channel] = \
	#  		old_weights[:, : filter_index * params_per_input_channel]
	#  	new_weights[:, filter_index * params_per_input_channel :] = \
	#  		old_weights[:, (filter_index + 1) * params_per_input_channel :]
	 	
	#  	# new_linear_layer.bias.data = old_linear_layer.bias.data

	#  	# new_linear_layer.weight.data = torch.from_numpy(new_weights).cuda()

	# 	classifier = torch.nn.Sequential(
	# 		*(replace_layers(model.classifier, i, [layer_index], \
	# 			[new_linear_layer]) for i, _ in enumerate(model.classifier)))

	# 	del model.classifier
	# 	del next_conv
	# 	del conv
	# 	model.classifier = classifier

	return model

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.train()

	t0 = time.time()
	model = prune_conv_layer(model, 28, 10)
	print "The prunning took", time.time() - t0