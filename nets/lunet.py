# 常用资源库
import pandas as pd
import numpy as np
EPS = 1e-9#
import os,glob,numbers
# 图像处理
import math,cv2,random
from PIL import Image, ImageFile, ImageOps, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

import torch
import torch.nn as nn
import torch.nn.functional as F


#start#
def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				out='dis', activation=swish, conv=nn.Conv2d, 
				):#'frelu',nn.ReLU(inplace=False),sinlu
		super(BasicConv2d, self).__init__()
		if not isinstance(kernel_size, tuple):
			if dilation>1:
				padding = dilation*(kernel_size//2)	#AtrousConv2d
			elif kernel_size==stride:
				padding=0
			else:
				padding = kernel_size//2			#BasicConv2d

		self.c = conv(in_channels, out_channels, 
			kernel_size=kernel_size, stride=stride, 
			padding=padding, dilation=dilation, bias=bias)
		self.b = nn.BatchNorm2d(out_channels, momentum=0.01) if bn else nn.Identity()

		self.o = nn.Identity()
		drop_prob=0.15
		# self.o = DisOut(drop_prob=drop_prob)#
		self.o = nn.Dropout2d(p=drop_prob, inplace=False) 

		if activation=='frelu':
			self.a = FReLU(out_channels)
		elif activation is None:
			self.a = nn.Identity()
		else:
			self.a = activation

	def forward(self, x):
		x = self.c(x)# x = torch.clamp_max(x, max=99)
		# print('c:', x.max().item())
		x = self.o(x)
		# print('o:', x.max().item())
		x = self.b(x)
		# print('b:', x.max().item())
		x = self.a(x)
		# print('a:', x.max().item())
		return x

class ConvBlock(torch.nn.Module):
	attention=None
	MyConv = BasicConv2d
	def __init__(self, inp_c, out_c, ksize=3, shortcut=False, pool=True):
		super(ConvBlock, self).__init__()
		self.shortcut = nn.Sequential(nn.Conv2d(inp_c, out_c, kernel_size=1), nn.BatchNorm2d(out_c))
		pad = (ksize - 1) // 2

		if pool: self.pool = nn.MaxPool2d(kernel_size=2)
		else: self.pool = False

		block = []
		block.append(self.MyConv(inp_c, out_c, kernel_size=ksize, padding=pad))

		if self.attention=='ppolar':
			# print('ppolar')
			block.append(ParallelPolarizedSelfAttention(out_c))
		elif self.attention=='spolar':
			# print('spolar')
			block.append(SequentialPolarizedSelfAttention(out_c))
		elif self.attention=='siamam':
			# print('siamam')
			block.append(simam_module(out_c))
		# else:
		# 	print(self.attention)
		block.append(self.MyConv(out_c, out_c, kernel_size=ksize, padding=pad))
		self.block = nn.Sequential(*block)
	def forward(self, x):
		if self.pool: x = self.pool(x)
		out = self.block(x)
		return swish(out + self.shortcut(x))

# 输出层 & 下采样
class OutSigmoid(nn.Module):
	def __init__(self, inp_planes, out_planes=1, out_c=8):
		super(OutSigmoid, self).__init__()
		self.cls = nn.Sequential(
			nn.Conv2d(in_channels=inp_planes, out_channels=out_c, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_c),
			nn.Conv2d(in_channels=out_c, out_channels=1, kernel_size=1, bias=True),
			nn.Sigmoid()
		)
	def forward(self, x):
		return self.cls(x)

class UpsampleBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, up_mode='transp_conv'):
		super(UpsampleBlock, self).__init__()
		block = []
		if up_mode == 'transp_conv':
			block.append(nn.ConvTranspose2d(inp_c, out_c, kernel_size=2, stride=2))
		elif up_mode == 'up_conv':
			block.append(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False))
			block.append(nn.Conv2d(inp_c, out_c, kernel_size=1))
		else:
			raise Exception('Upsampling mode not supported')

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class ConvBridgeBlock(torch.nn.Module):
	def __init__(self, out_c, ksize=3):
		super(ConvBridgeBlock, self).__init__()
		pad = (ksize - 1) // 2
		block=[]

		block.append(nn.Conv2d(out_c, out_c, kernel_size=ksize, padding=pad))
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(out_c))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class UpConvBlock(torch.nn.Module):
	def __init__(self, inp_c, out_c, ksize=3, up_mode='up_conv', conv_bridge=False, shortcut=False):
		super(UpConvBlock, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = UpsampleBlock(inp_c, out_c, up_mode=up_mode)
		self.conv_layer = ConvBlock(2 * out_c, out_c, ksize=ksize, shortcut=shortcut, pool=False)
		if self.conv_bridge:
			self.conv_bridge_layer = ConvBridgeBlock(out_c, ksize=ksize)

	def forward(self, x, skip):
		up = self.up_layer(x)
		if self.conv_bridge:
			out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
		else:
			out = torch.cat([up, skip], dim=1)
		out = self.conv_layer(out)
		return out

class LUNet(nn.Module):
	__name__ = 'lunet'
	use_render = False
	def __init__(self, inp_c=1, n_classes=1, layers=(32,32,32,32,32)):
		super(LUNet, self).__init__()
		self.num_features = layers[-1]

		self.__name__ = 'u{}x{}'.format(len(layers), layers[0])
		self.n_classes = n_classes
		self.first = BasicConv2d(inp_c, layers[0])

		self.down_path = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = ConvBlock(inp_c=layers[i], out_c=layers[i + 1], pool=True)
			self.down_path.append(block)

		self.up_path = nn.ModuleList()
		reversed_layers = list(reversed(layers))
		for i in range(len(layers) - 1):
			block = UpConvBlock(inp_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.up_path.append(block)

		self.out = OutSigmoid(layers[0], n_classes)

	def forward(self, x):
		x = self.first(x)
		down_activations = []
		for i, down in enumerate(self.down_path):
			down_activations.append(x)
			# print(x.shape)
			x = down(x)
		down_activations.reverse()

		for i, up in enumerate(self.up_path):
			x = up(x, down_activations[i])

		x = self.out(x)
		return x

def lunet(**args):
	net = LUNet(**args)
	net.__name__ = 'lunet'
	return net
#end#



if __name__ == '__main__':
	import time

	net = lunet()


	x = torch.rand(2,1,64,64)

	st = time.time()
	y = net(x)
	print('time:', time.time()-st)

	print('pred:', y.shape, y.min().item(), y.max().item())
	# print(net.__name__, y['loss'])

	# plot(net.emb)
	print('Params model:',sum(p.numel() for p in net.parameters() if p.requires_grad))
