import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append('.')
sys.path.append('..')

from nets import *
from utils import *

class MlpNorm(nn.Module):
	def __init__(self, dim_inp=256, dim_out=64):
		super(MlpNorm, self).__init__()
		dim_mid = min(dim_inp, dim_out)#max(dim_inp, dim_out)//2
		# hidden layers

		self.linear_hidden = nn.Sequential(
			nn.Linear(dim_inp, dim_mid),
			nn.Dropout(p=0.2),
			nn.LeakyReLU(),
			nn.Linear(dim_mid, dim_out),
			nn.BatchNorm1d(dim_out)
		)

	def forward(self, x):
		x = self.linear_hidden(x)
		return F.normalize(x, p=2, dim=-1)


#start#
from torchvision.utils import make_grid
def swish(x):
	# return x * torch.sigmoid(x)   #计算复杂
	return x * F.relu6(x+3)/6       #计算简单

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, 
				stride=1, padding=1, dilation=1, groups=1, bias=False, bn=True, 
				activation=swish, conv=nn.Conv2d, 
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

		if activation is None:
			self.a = nn.Sequential()
		else:
			self.a = activation
		
		self.b = nn.BatchNorm2d(out_channels) if bn else nn.Sequential()
		self.o = nn.Dropout2d(p=0.15)#DisOut(p=.15)#

	def forward(self, x):
		x = self.c(x)
		x = self.b(x)
		x = self.o(x)
		x = self.a(x)
		return x

class BottleNeck(torch.nn.Module):
	MyConv = BasicConv2d
	def __init__(self, in_c, out_c, stride=1, out='dis', **args):
		super(BottleNeck, self).__init__()
		if in_c!=out_c:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False), 
				nn.BatchNorm2d(out_c)
				)
		else:
			self.shortcut = nn.Sequential()
		self.conv1 = self.MyConv(in_c, out_c, 3, padding=1)
		self.conv2 = self.MyConv(out_c, out_c, 3, padding=1, activation=None)
		# self.o = nn.Dropout2d(p=0.15)#DisOut(p=.15)#
	def forward(self, x):
		out = self.conv2(self.conv1(x))
		# out = self.o(out)
		return swish(out + self.shortcut(x))

class BlockUnpool(torch.nn.Module):
	def __init__(self, in_c, out_c, up_mode='transp_conv'):
		super(BlockUnpool, self).__init__()
		block = []
		block.append(nn.Conv2d(in_c, in_c, kernel_size=1))
		block.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))
		block.append(nn.Conv2d(out_c, out_c, kernel_size=1))
		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class BlockBridge(torch.nn.Module):
	def __init__(self, channels, k_sz=3):
		super(BlockBridge, self).__init__()
		pad = (k_sz - 1) // 2
		block=[]

		block.append(nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad, bias=True))
		block.append(nn.LeakyReLU())
		block.append(nn.BatchNorm2d(channels))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out

class BlockUpsample(torch.nn.Module):
	def __init__(self, in_c, out_c, k_sz=3, conv_bridge=True):
		super(BlockUpsample, self).__init__()
		self.conv_bridge = conv_bridge

		self.up_layer = BlockUnpool(in_c, out_c)
		self.conv_layer = BottleNeck(2 * out_c, out_c)
		if self.conv_bridge:
			self.conv_bridge_layer = BlockBridge(out_c)

	def forward(self, x, skip):
		up = self.up_layer(x)
		if self.conv_bridge:
			out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
		else:
			out = torch.cat([up, skip], dim=1)
		out = self.conv_layer(out)
		return out

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

class ChannelAttention(nn.Module):
	def __init__(self, in_planes, ratio=16):
		super(ChannelAttention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
		self.relu1 = nn.ReLU()
		self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
		max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
		out = avg_out + max_out
		return self.sigmoid(out)

class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=7):
		super(SpatialAttention, self).__init__()
		assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
		padding = 3 if kernel_size == 7 else 1
		self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		x = torch.cat([avg_out, max_out], dim=1)
		x = self.conv1(x)
		return self.sigmoid(x)

class CBAM(nn.Module):
	def __init__(self, in_channels=32):
		super(CBAM, self).__init__()
		self.ch_att = ChannelAttention(in_channels)
		self.sp_att = SpatialAttention(3)
	def forward(self, x):
		x = self.ch_att(x)*x
		x = self.sp_att(x)*x
		return x

class MFISA(nn.Module):# Match filter inspired space attention，考虑对方差进行平滑性监督
	def __init__(self, in_channels=32):
		super(MFISA, self).__init__()
		mid_channels = in_channels//2
		self.filt_max = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3,1,1),
			nn.Conv2d(mid_channels, 1, 3,1,1),
			nn.BatchNorm2d(1)
		)
		self.filt_std = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3,1,1),
			nn.Conv2d(mid_channels, 1, 3,1,1),
			nn.BatchNorm2d(1)
		)
		self.filt_out = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3,1,1),
			nn.Conv2d(mid_channels, 1, 3,1,1),
			nn.BatchNorm2d(1)
		)
		self.filt_att = nn.Sequential(
			nn.Conv2d(1, mid_channels, 3,1,1),
			nn.Conv2d(mid_channels, 1, 3,1,1),
			nn.Sigmoid()
		)
		
	def forward(self, res):
		vmin, _ = torch.min(res, dim=1, keepdim=True)
		vmax, _ = torch.max(res, dim=1, keepdim=True)
		vp2p = vmax - vmin
		vstd = torch.std(res, dim=1, keepdim=True)
		
		# print(x.shape, m.shape, i.shape)
		att = self.filt_max(vp2p) + self.filt_std(vstd) + self.filt_out(vp2p + vstd)
		return self.filt_att(F.leaky_relu(att))*res

# class MFICA(nn.Module):# Match filter inspired channel attention
# 	def __init__(self, in_channels=32):
# 		super(MFICA, self).__init__()
# 		mid_channels = 8
# 		self.filt_pos = nn.Sequential(
# 			nn.Conv2d(1, 8, 3,1,1),
# 			nn.Conv2d(8, mid_channels, 3,1,1),
# 			nn.Conv2d(mid_channels, in_channels, 3,1,1),
# 			nn.BatchNorm2d(in_channels)
# 		)
# 		self.filt_neg = nn.Sequential(
# 			nn.Conv2d(1, 8, 3,1,1),
# 			nn.Conv2d(8, mid_channels, 3,1,1),
# 			nn.Conv2d(mid_channels, in_channels, 3,1,1),
# 			nn.BatchNorm2d(in_channels)
# 		)
# 		self.filt_std = nn.Sequential(
# 			nn.Conv2d(1, 8, 3,1,1),
# 			nn.Conv2d(8, mid_channels, 3,1,1),
# 			nn.Conv2d(mid_channels, in_channels, 3,1,1),
# 			nn.BatchNorm2d(in_channels)
# 		)
# 		self.filt_att = nn.Sequential(
# 			nn.Conv2d(in_channels,   in_channels*4, 1,1,0),
# 			nn.Conv2d(in_channels*4, in_channels*4, 1,1,0),
# 			nn.Conv2d(in_channels*4, in_channels,   1,1,0),
# 			# nn.Sigmoid()
# 			nn.Softmax(dim=1)
# 		)
		
# 	def forward(self, res):
# 		idxp = torch.argmax(res, dim=1, keepdim=True).type(torch.float32)
# 		idxn = torch.argmin(res, dim=1, keepdim=True).type(torch.float32)
# 		vstd = torch.argmax(torch.abs(res) , dim=1, keepdim=True).type(torch.float32)
# 		# vstd = torch.std(res, dim=1, keepdim=True)
# 		idxp = self.filt_pos(idxp)
# 		idxn = self.filt_neg(idxn)
# 		vstd = self.filt_std(vstd)
# 		# print(idxp.shape, idxn.shape, vstd.shape)
# 		att = self.filt_att(idxp+idxn+vstd)
# 		# print(att.shape, res.shape)
# 		return att*res

class MFAU(nn.Module):#Match filter inspired Attention unet
	__name__ = 'mfau'
	def __init__(self, in_channels=1, num_con=32, layers=(32,32,32,32,32), **args):
		super(MFAU, self).__init__()
		reversed_layers = list(reversed(layers))
		self.first = BottleNeck(in_c=in_channels, out_c=layers[0])
		self.pool = nn.Conv2d(layers[0], layers[0], kernel_size=3, stride=2, padding=1)

		self.projector = MlpNorm(layers[0], num_con)
		self.predictor = MlpNorm(num_con, num_con)

		self.attenten = nn.ModuleList()
		self.attentde = nn.ModuleList()

		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()
		self.dscoders = nn.ModuleList()
		for i in range(len(layers) - 1):
			self.encoders.append(BottleNeck(in_c=layers[i], out_c=layers[i + 1]))
			
			self.attenten.append(MFISA(layers[i + 1]))
			self.attentde.append(MFISA(layers[i + 1]))
			
			block = BlockUpsample(in_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.decoders.append(block)
			
			self.dscoders.append(OutSigmoid(layers[i + 1]))

		self.final = OutSigmoid(layers[0])
		# self.ten = TextureExtractionNet()

	def forward(self, x0):
		# x0 = self.ten(x0)
		# x0 = 1-x0
		# print('forward:', x.shape)
		x = self.first(x0)
		self.shallow = x

		down_activations = []
		for i, down in enumerate(self.encoders):
			down_activations.append(x)
			x = down(self.pool(x))
			x = self.attenten[i](x)
			# x = self.attentde[i](x)

		down_activations.reverse()
		auxs = []
		for i, up in enumerate(self.decoders):
			x = up(x, down_activations[i])
			x = self.attentde[i](x)

			y = self.dscoders[2-i](x)
			y = F.interpolate(y, size=x0.shape[-2:], mode='bilinear', align_corners=False)
			auxs.append(y)
			
		self.feat = self.shallow+x
		y = self.final(self.feat)
		self.pred = y
		auxs.append(y)
		auxs.reverse()
		return auxs

def mfau(**args):
	return MFAU(**args)

class TVLoss(nn.Module):
	def __init__(self):
		super(TVLoss, self).__init__()

	def forward(self, x):
		batch_size = x.size()[0]
		h_x = x.size()[2]
		w_x = x.size()[3]
		count_h = self.tensor_size(x[:, :, 1:, :])
		count_w = self.tensor_size(x[:, :, :, 1:])
		h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
		w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
		return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

	@staticmethod
	def tensor_size(t):
		return t.size()[1] * t.size()[2] * t.size()[3]

class MatchGroup(nn.Module):
	channels=12
	def __init__(self, sigma=3, **args):#13,17
		super(MatchGroup, self).__init__()
		self.tv = TVLoss()
		ksize = 2*sigma+1
		self.pad = sigma
		self.kernel = nn.Parameter(torch.rand(1,1,ksize,ksize), requires_grad=True)
		# https://blog.csdn.net/bxdzyhx/article/details/112729725
		for i,theta in enumerate(torch.arange(0, np.pi, np.pi/self.channels)):
			grid = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0]])
			grid = F.affine_grid(grid[None, ...], (1,1,ksize,ksize), align_corners=True)
			self.register_buffer('grid'+str(i), grid.type(torch.float32))

	def plot(self):
		res = []
		for i in range(self.channels):
			m = F.grid_sample(self.kernel, eval('self.grid'+str(i)), align_corners=True)
			res.append(m)
		imag = torch.cat(res, dim=0)
		imag = make_grid(imag, nrow=4, normalize=True)
		# print('plot fcn:', imag.shape, imag.min().item(), imag.max().item())
		return imag.unsqueeze(0)
	def regular(self, rot=False, oth=True, std=False, tvl=False):
		losDMF = {'rot':0, 'oth':0, 'std':0, 'tvl':self.losTV}
		if oth:
			losOth = 0
			for i in range(self.channels):
				raw = self.kernel
				oth = F.grid_sample(raw, eval('self.grid'+str(6)), align_corners=True)
				resRaw = torch.norm(F.conv2d(raw, raw, bias=None))
				resOth = torch.norm(F.conv2d(raw, oth, bias=None))
				losOth = losOth + resOth / (resRaw+resOth+1e-3)#参考dice的形式
				# losOth = losOth + resOth / (resRaw+1e-3)#参考dice的形式
			losDMF['oth'] = losOth/self.channels
			# print('los-oth:', losOth)
		if std:                                                                                                                                                                                                                                                                                                                                                         
			losSTD = torch.norm(self.kernel.std()-1)
			losDMF['std'] = losSTD
		return losDMF
		# return sum(u.regular() for u in self.kernel)/len(self.kernel)
	losTV = 0
	def forward(self, x):
		res = []
		self.losTV = 0
		for i in range(self.channels):
			m = F.grid_sample(self.kernel, eval('self.grid'+str(i)), align_corners=True)
			r = F.conv2d(x, m, bias=None, padding=self.pad)
			# r = F.leaky_relu(r)
			res.append(r)
		y = torch.cat(res, dim=1)
		self.losTV = self.tv(y)
		return y

class MatchSCIRD(nn.Module):
	channels=12
	__name__ = 'scird'
	# ks = [-0.1, -0.05, 0, 0.05, 0.1]
	def __init__(self, sigma=5, index=0):#13,17
		super(MatchSCIRD, self).__init__()
		self.pad = sigma
		self.ksize = sigma*2+1
		# self.k_dim = self.ks[index]

		self.sigma = nn.Parameter(torch.randn(1), requires_grad=True)
		self.gamma = nn.Parameter(torch.randn(1), requires_grad=True)
		self.ratio = nn.Parameter(torch.randn(1), requires_grad=True)

		# 转成Parameter才能自动放到GPU上
		x,y = torch.meshgrid(torch.arange(0, self.ksize, 1), torch.arange(0, self.ksize, 1))
		gridx = x - self.ksize // 2
		gridy = y - self.ksize // 2
		thetas = torch.arange(0, math.pi, math.pi/self.channels)
		for i,theta in enumerate(thetas):
			cos, sin = torch.cos(theta), torch.sin(theta)
			gdx = gridx * cos + gridy * sin
			gdy = gridy * cos - gridx * sin 

			self.register_buffer('gdx'+str(i), gdx)
			self.register_buffer('gdy'+str(i), gdy)

	def plot(self):
		kernels = self.make_kernels()
		imag = torch.cat(kernels, dim=0)
		imag = make_grid(imag, nrow=4, normalize=True)
		# print('plot dmf:', imag.shape, imag.min().item(), imag.max().item())
		
		gamma = 1.5 + torch.tanh(self.gamma)
		sigma = 1.5 + torch.tanh(self.sigma)
		print('SCIRD-GAMMA&SIGMA:', gamma.item(), sigma.item())
		return imag.unsqueeze(0)
	def regular(self, rot=True, oth=True, std=True, tvl=True):
		return {'rot':0, 'oth':0, 'std':0, 'tvl':0}
	def make_kernels(self):
		gamma = 1.5 + torch.tanh(self.gamma)
		sigma = 1.5 + torch.tanh(self.sigma)
		sigma2 = sigma**2
		ratio = 0.3 * torch.tanh(self.ratio)

		kernels = []
		for i in range(self.channels):
			gdx = eval('self.gdx'+str(i))
			gdy = eval('self.gdy'+str(i))

			# %SCIRD-TS filter
			gdy = (gdy + ratio*gdx**2)**2
			kernel = torch.exp(-(gdx**2)/(2*gamma**2)-gdy/(2*sigma2)) * (gdy/(sigma**4) - 1/(sigma2))

			kernels.append(kernel.view(1,-1,self.ksize,self.ksize))
		return kernels

	def forward(self, x):
		res = []
		for kernel in self.make_kernels():
			res.append(F.conv2d(x, kernel, padding=self.pad))
		y = torch.cat(res, dim=1)
		return y

class MatchMS(nn.Module):# Multi-Scale Match
	block=MatchGroup
	def __init__(self, sigmas=[1,2,3,4,5]):
		super(MatchMS, self).__init__()
		self.channels = len(sigmas)#*MatchGroup.channels
		print('Deep Matched Filtering Channels:', self.channels)
		self.groups = nn.ModuleList()
		for i,sigma in enumerate(sigmas):
			group = self.block(sigma=sigma, index=i)
			self.groups.append(group)
		self.out = OutSigmoid(2)
		self.idx = nn.Sequential(
			nn.Conv2d(1,1,3,1,1),
			nn.Conv2d(1,1,3,1,1)
		)
	def regular(self, **args):
		losDMF = {'rot':0, 'oth':0, 'std':0, 'tvl':0}
		for g in self.groups:
			los = g.regular(**args)
			for key in los.keys():
				losDMF[key] = losDMF[key] + los[key]/len(self.groups)
		return losDMF
	def forward(self, x):
		res = sum([group(x) for group in self.groups])
		# print(res.shape)
		amin, aidx = torch.min(res, dim=1, keepdim=True)
		amax, aidx = torch.max(res, dim=1, keepdim=True)
		amp = amax - amin
		std = torch.std(res, dim=1, keepdim=True)
		# print('MatcMS:', outputs.shape)
		return self.out(torch.cat([amp, std], dim=1))

	def plot(self, root='./'):
		imgs = []
		for i,m in enumerate(self.groups):
			img = m.plot()
			# print(img.shape)
			img = F.interpolate(img, size=(90,120), mode='nearest')
			# img = torchvision.utils.make_grid(img, nrow=4, padding=8)
			if root is not None:
				torchvision.utils.save_image(img, root+'/mf{}.png'.format(i))
			imgs.append(img)
		if root is None:
			plt.subplot(151),plt.imshow(imgs[0][0][0].data.numpy())
			plt.subplot(152),plt.imshow(imgs[1][0][0].data.numpy())
			plt.subplot(153),plt.imshow(imgs[2][0][0].data.numpy())
			plt.subplot(154),plt.imshow(imgs[3][0][0].data.numpy())
			plt.subplot(155),plt.imshow(imgs[4][0][0].data.numpy())
			plt.show()

class AttentionPool(nn.Module):
	def __init__(self):
		super(AttentionPool, self).__init__()
		self.pool = nn.Sequential(
			nn.Conv2d(1,1,3,1,1),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Conv2d(1,1,3,1,1),
			nn.BatchNorm2d(1)
		)
		self.conv = nn.Sequential(
			nn.Conv2d(1,1,3,1,1),
			nn.Conv2d(1,1,3,2,1),
			nn.Conv2d(1,1,3,1,1),
			nn.BatchNorm2d(1)
		)
		self.restore = nn.Sequential(
			nn.Conv2d(1,1,3,1,1),
			nn.UpsamplingBilinear2d(scale_factor=2),
			nn.Conv2d(1,1,3,1,1),
			nn.BatchNorm2d(1),
			nn.Conv2d(1,1,3,1,1),
			nn.Sigmoid()
		)
	def regular(self):
		return self.loss
	def forward(self, x):
		att = torch.sigmoid(self.pool(x)+self.conv(x))
		restore = self.restore(att)
		self.loss = F.mse_loss(x, restore)
		return att

class MFGU(nn.Module):#Match filter inspired Attention unet
	__name__ = 'mfgu'
	def __init__(self, in_channels=1, type_seg='gauss', num_con=32, layers=(32,32,32,32,32), **args):
		super(MFGU, self).__init__()
		if type_seg=='gauss':
			MatchMS.block = MatchGroup
		elif type_seg=='scird':
			MatchMS.block = MatchSCIRD
		else:
			raise NotImplementedError('No this kernel')
		self.__name__ += type_seg

		reversed_layers = list(reversed(layers))
		self.first = BottleNeck(in_c=in_channels, out_c=layers[0])
		self.pool = nn.Conv2d(layers[0], layers[0], kernel_size=3, stride=2, padding=1)

		self.projector = MlpNorm(layers[0], num_con)
		self.predictor = MlpNorm(num_con, num_con)
		self.fcn = MatchMS()
		self.fcn_pool = nn.ModuleList()
		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()
		self.dscoders = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = BottleNeck(in_c=layers[i], out_c=layers[i + 1])
			self.encoders.append(block)
			
			block = BlockUpsample(in_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.decoders.append(block)
			
			block = OutSigmoid(layers[i + 1])
			self.dscoders.append(block)

			self.fcn_pool.append(AttentionPool())

		self.final = OutSigmoid(layers[0])

	def regular(self):
		return sum([p.regular() for p in self.fcn_pool])

	tmp={}
	def forward(self, x0):
		# print('forward:', x.shape)
		x = self.first(x0)
		self.shallow = x

		f = self.fcn(1-x0)
		self.tmp['dmf'] = f
		# print(x0.shape, f.shape)

		fs = [f]
		down_activations = []
		for i, down in enumerate(self.encoders):
			down_activations.append(x)
			x = down(self.pool(x))
			# fx = F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=False)
			fx = self.fcn_pool[i](fs[-1])
			# print(x.shape, fx.shape)
			fs.append(fx)
			x = x*fx

		down_activations.reverse()
		auxs = [f]
		for i, up in enumerate(self.decoders):
			# x = self.dropout(x)
			x = up(x, down_activations[i])
			# fx = F.interpolate(f, size=x.shape[-2:], mode='bilinear', align_corners=False)
			fx = fs[2-i]
			x = x*fx

			y = self.dscoders[2-i](x)
			y = F.interpolate(y, size=x0.shape[-2:], mode='bilinear', align_corners=False)
			auxs.append(y)
			
		self.feat = self.shallow+x
		y = self.final(self.feat)
		self.pred = y
		auxs.append(y)
		auxs.reverse()
		return auxs

def mfgu(**args):
	return MFGU(**args)


class MFU(nn.Module):#Match filter inspired Attention unet
	__name__ = 'mfu'
	def __init__(self, in_channels=1, type_seg='gauss', num_con=32, layers=(32,32,32,32,32), ATT_BLOCK=MFISA, **args):
		super(MFU, self).__init__()
		if type_seg=='gauss':
			MatchMS.block = MatchGroup
		else:
			raise NotImplementedError('No this kernel')
		self.__name__ += type_seg

		reversed_layers = list(reversed(layers))
		self.first = BottleNeck(in_c=in_channels, out_c=layers[0])
		self.pool = nn.Conv2d(layers[0], layers[0], kernel_size=3, stride=2, padding=1)
		self.fcn = MatchMS()

		self.projector = MlpNorm(layers[0], num_con)
		self.predictor = MlpNorm(num_con, num_con)

		self.fcn_pool = nn.ModuleList()
		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()
		self.attenten = nn.ModuleList()
		self.attentde = nn.ModuleList()
		self.dscoders = nn.ModuleList()
		# self.tacoders = nn.ModuleList()
		for i in range(len(layers) - 1):
			block = BottleNeck(in_c=layers[i], out_c=layers[i + 1])
			self.encoders.append(block)
			
			# self.attenten.append(MFISA(layers[i + 1]))
			# self.attentde.append(MFISA(layers[i + 1]))
			self.attenten.append(ATT_BLOCK(layers[i + 1]))
			self.attentde.append(ATT_BLOCK(layers[i + 1]))
			
			block = BlockUpsample(in_c=reversed_layers[i], out_c=reversed_layers[i + 1])
			self.decoders.append(block)
			
			# self.tacoders.append(nn.Conv2d(layers[i+1], layers[i+1]*self.aux_up[3-i]**2, kernel_size=1, groups=layers[i+1]))
			self.dscoders.append(OutSigmoid(layers[i+1]))

			self.fcn_pool.append(AttentionPool())

		self.final = OutSigmoid(layers[0])

	def regular(self, x=None, **args):
		return sum([p.regular() for p in self.fcn_pool])

	tmp={}
	aux_up = [8,4,2,1]
	aux_in = [4,8,16,32]
	def forward(self, x0):
		h,w = x0.shape[-2:]
		# print('forward:', x.shape)
		x = self.first(x0)
		self.shallow = x

		f = self.fcn(1-x0)
		self.tmp['dmf'] = f

		fs = [f]
		down_activations = []
		for i, down in enumerate(self.encoders):
			down_activations.append(x)
			fx = self.fcn_pool[i](fs[-1])
			fs.append(fx)
			x = down(self.pool(x)*fx)
			x = self.attenten[i](x)

		down_activations.reverse()
		auxs = [f]
		for i, up in enumerate(self.decoders):
			# x = self.dropout(x)
			x = up(x, down_activations[i])
			x = self.attentde[i](x)

			y = x
			# print(y.shape)
			y = self.dscoders[3-i](y)
			y = F.interpolate(y, size=x0.shape[-2:], mode='bilinear', align_corners=False)
			auxs.append(y)
			
		self.feat = self.shallow+x
		y = self.final(self.feat)
		self.pred = y
		auxs.append(y)
		auxs.reverse()
		return auxs

# def mfu(**args):
# 	return MFU(**args)
def mfc(**args):
	net = MFU(**args, ATT_BLOCK=CBAM)
	net.__name__ = 'mfc'
	return net
def mfu(**args):
	net = MFU(**args, ATT_BLOCK=MFISA)
	net.__name__ = 'mfu'
	return net
#end#



if __name__ == '__main__':
	# x = torch.rand(8,32,128,128)
	# net = MFISA()
	# net = MFICA()
	
	x = torch.rand(2,1,128,128)
	# net = mfau()
	# net = mfgu()
	net = mfu()
	# net = mfc()
	print(net.__name__)
	print('params:', sum(p.numel() for p in net.parameters() if p.requires_grad))

	# net.eval()
	import time
	st = time.time()
	ys = net(x)
	# print(net.__name__, net.feat.shape)
	print('Time:', time.time() - st)
	# Time: 0.44896960258483887 
	# print('Loss:', net.regular())
	if hasattr(net, 'regular'):
		print('regular:', net.regular())


	if isinstance(ys, (list, tuple)):
		for y in ys:
			print(y.shape, y.min().item(), y.max().item())
	else:
		print(ys.shape, ys.min().item(), ys.max().item())

	del net.projector, net.predictor
	print('Params:',sum(p.numel() for p in net.parameters() if p.requires_grad))
	# 主要的参数都在UNet上面，后面把UNet参数调小就好了
	# print('Params fcn:',sum(p.numel() for p in net.fcn.parameters() if p.requires_grad))
	# print('Params ten:',sum(p.numel() for p in net.ten.parameters() if p.requires_grad))
	# print('Params seg:',sum(p.numel() for p in net.seg.parameters() if p.requires_grad))

	# if hasattr(net, 'fcn') and hasattr(net.fcn, 'plot'):
	# 	net.fcn.plot(root=None)
