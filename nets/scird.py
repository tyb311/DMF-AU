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

class MatchGabor(nn.Module):
	channels=12
	__name__ = 'MatchGabor'
	# ks = [-0.1, -0.05, 0, 0.05, 0.1]
	def __init__(self, sigma=5, index=0):#13,17
		super(MatchGabor, self).__init__()
		self.pad = sigma
		self.ksize = sigma*2+1
		# self.k_dim = self.ks[index]

		self.lambd = nn.Parameter(torch.rand(1))
		self.phi = nn.Parameter(torch.randn(1) * 0.02)

		self.sigma = nn.Parameter(torch.randn(1) * 1.0, requires_grad=True)
		self.gamma = nn.Parameter(torch.randn(1) * 0.5, requires_grad=True)

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
	def regular(self, rot=True, oth=True, std=True, axi=True):
		return {'rot':0, 'oth':0, 'std':0, 'axi':0}
	def make_kernels(self):
		# gamma = .8 + torch.tanh(self.gamma)
		# sigma = 0.1 + torch.sigmoid(self.sigma) * 0.4
		# lambd = 0.001 + torch.sigmoid(self.lambd) * 0.999
		gamma = self.gamma
		sigma = torch.abs(self.sigma)
		lambd = torch.abs(self.lambd)

		kernels = []
		for i in range(self.channels):
			gdx = eval('self.gdx'+str(i))
			gdy = eval('self.gdy'+str(i))

			kernel = torch.exp(-0.5 * (gdx**2 + (gamma*gdy)**2) / sigma**2)
			kernel = kernel * torch.cos((2.0 * math.pi * (gdx / lambd)) +  self.phi)

			kernels.append(kernel.view(1,-1,self.ksize,self.ksize))
		return kernels

	def forward(self, x):
		res = []
		for kernel in self.make_kernels():
			res.append(F.conv2d(x, kernel, padding=self.pad))
		y = torch.cat(res, dim=1)
		# y = torch.max(y, dim=1, keepdim=True)[0]
		_y = torch.max(y, dim=1, keepdim=True)[0]
		y = torch.cat([y,_y], dim=1)
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
	def regular(self, rot=True, oth=True, std=True, axi=True):
		return {'rot':0, 'oth':0, 'std':0, 'axi':0}
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
		# y = torch.max(y, dim=1, keepdim=True)[0]
		_y = torch.max(y, dim=1, keepdim=True)[0]
		y = torch.cat([y,_y], dim=1)
		return y

class MatchGroup(nn.Module):
	channels=12
	def __init__(self, sigma=3, **args):#13,17
		super(MatchGroup, self).__init__()
		ksize = 2*sigma+1
		self.pad = sigma
		self.kernel = nn.Parameter(torch.rand(1,1,ksize,ksize), requires_grad=True)
		# https://blog.csdn.net/bxdzyhx/article/details/112729725
		for i,theta in enumerate(torch.arange(0, np.pi, np.pi/self.channels)):
			grid = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0]])
			grid = F.affine_grid(grid[None, ...], (1,1,ksize,ksize), align_corners=True)
			self.register_buffer('grid'+str(i), grid.type(torch.float32))

	def plot(self):
		rs = []
		for i in range(self.channels):
			m = F.grid_sample(self.kernel, eval('self.grid'+str(i)), align_corners=True)
			rs.append(m)
		imag = torch.cat(rs, dim=0)
		imag = make_grid(imag, nrow=4, normalize=True)
		# print('plot dmf:', imag.shape, imag.min().item(), imag.max().item())
		return imag.unsqueeze(0)
	def regular(self, rot=False, oth=True, std=False, axi=False):
		losDMF = {'rot':0, 'oth':0, 'std':0, 'axi':0}
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
	def forward(self, x):
		rs = []
		for i in range(self.channels):
			m = F.grid_sample(self.kernel, eval('self.grid'+str(i)), align_corners=True)
			r = F.conv2d(x, m, bias=None, padding=self.pad)
			r = F.leaky_relu(r)
			rs.append(r)
		y = torch.cat(rs, dim=1)
		y = torch.sort(y, dim=1)[0]
		# _y = torch.max(y, dim=1, keepdim=True)[0]
		_y = torch.std(y, dim=1, keepdim=True)#[0]
		y = torch.cat([y,_y], dim=1)
		# print('MatchGroup:', y.shape)
		return y

class MatchMS(nn.Module):# Multi-Scale Match
	def __init__(self, sigmas=[1,2,3,4,5], block=MatchGroup):
		super(MatchMS, self).__init__()
		self.channels = len(sigmas)#*MatchGroup.channels
		print('Deep Matched Filtering Channels:', self.channels)
		self.groups = nn.ModuleList()
		for i,sigma in enumerate(sigmas):
			# group = MatchGroup(sigma=sigma, index=i)
			# group = MatchGabor(sigma=sigma, index=i)
			# group = MatchSCIRD(sigma=sigma, index=i)
			group = block(sigma=sigma, index=i)
			self.groups.append(group)
	def regular(self, **args):
		losDMF = {'rot':0, 'oth':0, 'std':0, 'axi':0}
		for g in self.groups:
			los = g.regular(**args)
			for key in los.keys():
				losDMF[key] = losDMF[key] + los[key]/len(self.groups)
		return losDMF
	def forward(self, x):
		outputs = torch.cat([group(x) for group in self.groups], dim=1)
		# print('MatcMS:', outputs.shape)
		return outputs#self.out(outputs)
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

class DMFNet(torch.nn.Module):	## neural matched-filtering attention net.  DMFNet Net
	__name__ = 'dmf'
	def __init__(self, type_net='dmf', type_seg='res', num_con=32, filters=32):
		super(DMFNet, self).__init__()
		if type_seg=='scird':
			print('MatchSCIRD')
			self.fcn = MatchMS(block=MatchSCIRD)
		elif type_seg=='gabor':
			print('MatchGabor')
			self.fcn = MatchMS(block=MatchGabor)
		else:
			type_seg = 'gauss'
			print('MatchGroup')
			self.fcn = MatchMS(block=MatchGroup)
		print('@'*32, 'Match Filter Kernel:', type_seg)
		self.__name__ = 'dmf'+type_seg

		self.ten = TextureExtractionNet()
		# self.ten = nn.Conv2d(1,1,1,1,0)
		self.fusion = nn.Conv2d(self.fcn.channels*13, filters, kernel_size=1,stride=1,padding=0,bias=True)
		# self.fusion = nn.Conv2d(1, filters, kernel_size=1,stride=1,padding=0,bias=True), 
		# print('Channels=', channels)
		
		self.projector = MlpNorm(filters, num_con)
		self.predictor = MlpNorm(num_con, num_con)

		self.seg = LUNet(filters + 2, n_classes=1, layers=(filters,)*4)
		self.aux_ten = OutSigmoid(1)
		self.aux_dmf = OutSigmoid(self.fcn.channels*13)
		self.aux_fet = OutSigmoid(filters)
		# self.aux_dmf = nn.ModuleList([
		# 	OutSigmoid(13) for i in range(self.fcn.channels)
		# ])
	def regular(self, **args):
		return self.fcn.regular(**args)
	tmp = {}
	def forward(self, x0, **args):
		x0 = self.ten(x0)
		xt = 1-x0
		xd = self.fcn(xt)
		# self.feat = xr

		# print(xt.shape, xr.shape, x0.shape)
		xf = self.fusion(xd)
		self.feat = xf
		

		# st = time.time()
		o = self.seg(torch.cat([x0,xt,xf], dim=1))
		# o = self.seg(xr)
		# print('Time for SEG:', time.time()-st)
		# print(y.shape)
		if self.training:
			aux_ten = self.aux_ten(xt)
			aux_dmf = self.aux_dmf(xd)
			aux_fet = self.aux_fet(xf)
			self.pred = aux_dmf
			auxs = [o, aux_fet, aux_dmf, aux_ten]

			# xf = torch.chunk(xr, self.fcn.channels, dim=1)
			# # print('chunk:', len(xf))
			# for x,conv in zip(xf, self.aux_dmf):
			# 	aux = conv(x)
			# 	auxs.append(aux)
			return auxs
		return o

def dmf32(**args):	#deep matched filtering
	net = DMFNet(filters=32, **args)
	net.__name__ = 'dmf32'
	return net