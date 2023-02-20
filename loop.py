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
from utils import *

import time, tqdm, kornia#, socket
from torchvision.transforms import functional as f

import os, glob, sys, time, torch
from torch.optim import lr_scheduler
# from torch.cuda import amp
torch.set_printoptions(precision=3)

class GradUtil(object):
	def __init__(self, model, loss='ce', lr=0.01, wd=2e-4, root='.'):
		self.path_checkpoint = os.path.join(root, 'super_params.tar')
		if not os.path.exists(root):
			os.makedirs(root)

		self.lossName = loss
		self.criterion = get_loss(loss)
		params = filter(lambda p:p.requires_grad, model.parameters())
		self.optimizer = RAdamW(params=params, lr=lr, weight_decay=2e-4)
		self.scheduler = ReduceLR(name=loss, optimizer=self.optimizer,  
			mode='min', factor=0.7, patience=2, 
			verbose=True, threshold=0.0001, threshold_mode='rel', 
			cooldown=2, min_lr=1e-5, eps=1e-9)
		
	def isLrLowest(self, thresh=1e-5):
		return self.optimizer.param_groups[0]['lr']<thresh

	coff_ds = 0.5
	def calcGradient(self, criterion, outs, true, fov=None):
		lossSum = 0#torch.autograd.Variable(torch.tensor(0, dtype=torch.float32), requires_grad=True)
		if isinstance(outs, (list, tuple)):
			# ratio = 1/(1+len(outs))
			for i in range(len(outs)-1,0,-1):#第一个元素尺寸最大
				# print('输出形状：', outs[i].shape, true.shape)
				pred = outs[i][-true.shape[0]:,:true.shape[1],:true.shape[2],:true.shape[3]]
				loss = criterion(pred*fov, true)#, fov
				lossSum = lossSum + loss*self.coff_ds
			outs = outs[0]
		# print(outs.shape, true.shape)
		pred = outs[-true.shape[0]:,:true.shape[1],:true.shape[2],:true.shape[3]]
		lossSum = lossSum + criterion(pred*fov, true)#, fov
		self.total_loss += lossSum.item()
		return lossSum
		
	def backward_seg(self, pred, true, fov=None, model=None):
		los = self.calcGradient(self.criterion, pred, true, fov)
		del pred, true, fov
		return los

	total_loss = 0
	def update_scheduler(self, i=0):
		logStr = '\r{:03}# '.format(i)
		# losSum = 0
		logStr += '{}={:.4f},'.format(self.lossName, self.total_loss)
		print(logStr, end='')
		# self.callBackEarlyStopping(los=losSum)

		if isinstance(self.scheduler, ReduceLR):
			self.scheduler.step(self.total_loss)
		else:
			self.scheduler.step()
		self.total_loss = 0

from copy import deepcopy
class KerasBackend(object):
	bests = {'auc':0, 'iou':0, 'f1s':0, 'a':0}

	path_minlos = 'checkpoint_minloss.pt'
	path_metric = 'checkpoint_metrics.tar'
	paths = dict()
	logTxt = []
	isParallel = False
	def __init__(self, args, **kargs):
		super(KerasBackend, self).__init__()
		self.args = args
		# print('*'*32,'device')
		torch.manual_seed(311)
		self.device = torch.device('cpu')
		if torch.cuda.is_available():
			self.device = torch.device('cuda:0')  
			torch.cuda.empty_cache()
			torch.cuda.manual_seed_all(311)
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
			torch.backends.cudnn.enabled = True
			# Benchmark模式会提升计算速度，但计算中随机性使得每次网络前馈结果略有差异，
			# deterministic避免这种波动, 设置为False可以牺牲GPU提升精度

			current_device = torch.cuda.current_device()
			print(self.device, torch.cuda.get_device_name(current_device))
			for i in range(torch.cuda.device_count()):
				print("    {}:".format(i), torch.cuda.get_device_name(i))
		
	def save_weights(self, path):
		if not os.path.exists(self.root):
			os.mkdir(self.root)
		if self.isParallel:
			torch.save(self.model.module.state_dict(), path)
		else:
			# _model = deepcopy(self.model)
			# if hasattr(_model, 'uda'):
			# 	_model.uda = None
			torch.save(self.model.state_dict(), path)
		# print('save weigts to path:{}'.format(path))
	
	def load_weights(self, mode, desc=True):
		path = self.paths.get(mode, mode)#返回完全路径或者mode
		if mode=='los':
			path = self.path_minlos
		try:
			pt = torch.load(path, map_location=self.device)
			self.model.load_state_dict(pt, strict=False)#
			if self.isParallel:
				self.model = self.model.module
			if desc:print('Load from:', path)
			return True
		except Exception as e:
			print('Load wrong:', path)
			return False

	def init_weights(self):
		print('*'*32, 'Initial Weights--Ing!')
		for m in self.model.modules():
			if isinstance(m, nn.Conv2d) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.Linear) and  m.weight.requires_grad:
				torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
			elif isinstance(m, nn.BatchNorm2d) and  m.weight.requires_grad:
				torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
				torch.nn.init.constant_(m.bias.data, 0.0)

	def init_folders(self, dataset, losStr):
		timeStr = time.strftime("%m%d%H", time.localtime())
		if self.args.root=='':
			self.root = 'dmfau'
		else:
			self.root = self.args.root

		dataset.expCross = hasattr(dataset, 'expCross') and dataset.expCross
		if dataset.expCross: 
			self.path_metric = '{}/{}xcp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}xlos.pt'.format(self.root, dataset.dbname)
		else:
			self.path_metric = '{}/{}_cp.tar'.format(self.root, dataset.dbname)
			self.path_minlos = '{}/{}_los.pt'.format(self.root, dataset.dbname)
		print('Folder for experiment:', self.root)

		name_pt = dataset.dbname+'x' if dataset.expCross else dataset.dbname
		# print('Exec:', self.root)
		for key in self.bests.keys():
			self.paths[key] = '{}/{}-{}.pt'.format(self.root, name_pt, key)

	def compile(self, dataset, loss='fr', lr=0.01, **args): 
		#设置路径
		self.dataset = dataset
		self.init_folders(dataset, ''.join(loss))

		# 参数设置：反向传播、断点训练 
		self.gradUtil = GradUtil(model=self.model, loss=loss, lr=lr, root=self.root)
		if not self.load_weights(self.path_minlos):
			self.init_weights()
		print('Params total(KB):',sum(p.numel() for p in self.model.parameters()))#//245
		print('Params train(KB):',sum(p.numel() for p in self.model.parameters() if p.requires_grad))
			
		if self.isParallel:
			print('*'*32, 'Model Parallel')
			self.model = nn.DataParallel(self.model)  #, device_ids=[0,1,2,3]
			# self.model = nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))  
			# torch.cuda.set_device(self.device)
			# self.device = torch.device('cuda:0')
			self.model.to(self.device)
		else:
			print('*'*32, 'Model Serial')
			self.model = self.model.to(self.device) 

		try:
			self.bests = torch.load(self.path_metric)
			print('Metric Check point:', self.bests)
		except:
			print('Metric Check point none!') 
		
		self.gradUtil.criterion = self.gradUtil.criterion.to(self.device)
		
	def callBackModelCheckPoint(self, scores, lossItem=1e9):
		logStr = '\t'
		for mode in scores.keys():
			if scores[mode]>self.bests[mode]:
				logStr += '{}:{:6.4f}->{:6.4f},'.format(mode, self.bests[mode], scores[mode])
				self.bests[mode] = scores[mode]
				self.save_weights(self.paths[mode])   
		print(logStr)
		self.logTxt.append(logStr)
		torch.save(self.bests, self.path_metric)
		
	stop_counter=0
	stop_training = False
	best_loss = 9999
	isBestLoss = False
	def callBackEarlyStopping(self, los, epoch=0, patience=4):
		if los<self.best_loss:
			print('\tlos={:6.4f}->{:6.4f}'.format(self.best_loss, los))
			self.best_loss = los
			self.stop_counter=0
			# self.save_weights(self.path_minlos)
			self.isBestLoss = True
		else:
			print('\tlos={:6.4f}'.format(los))
			self.stop_counter+=1
			if self.stop_counter>patience and self.gradUtil.isLrLowest(1e-4) and epoch>100:
				self.stop_training = True
				print('EarlyStopp after:', patience)
	
		if self.isBestLoss:
			self.isBestLoss = False
			self.save_weights(self.path_minlos)

def make_trainable(model, val):
	for p in model.parameters():
		p.requires_grad = val

class KerasTorch(KerasBackend):
	evalEpochs = 3
	evalEpochs=3
	def __init__(self, model, **kargs):
		super(KerasTorch, self).__init__(**kargs)
		self.model = model
		# self.tv = TVLoss()
		self.dice = DiceLoss()
		self.bce = nn.BCELoss()

	def desc(self, key='my'):#, self.scheduler.get_lr()[0] 
		# print('Learing Rate:', self.optimizer.param_groups[0]['lr'])
		for n,m in self.model.named_parameters():
			if n.__contains__(key):
				print(n,m.detach().cpu().numpy())

	def fit(self, epochs=144):#现行验证，意义不大，把所有权重都验证要花不少时间
		self.stop_counter = 0
		self.stop_training = False            
		print('*'*32,'fitting:'+self.root) 
		# self.desc()
		time_fit_begin = time.time()
		for i in range(epochs):
			time_stamp = time.time()

			# 训练
			lossItem = self.train()
			logStr = '{:03}$ losSeg={:.4f}'.format(i, lossItem)
			print('\r'+logStr)
			self.logTxt.append(logStr)
			self.gradUtil.update_scheduler(i)

			# 验证
			if i>epochs*0.7:
				# if i>1 and i%self.evalEpochs==0:
				scores, lossItem = self.val()
				logStr = '{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f}'
				logStr = logStr.format(i, scores['auc'],scores['iou'],scores['f1s'])
				print('\r'+logStr)
				self.logTxt.append(logStr)
				self.callBackEarlyStopping(lossItem, i)
				self.callBackModelCheckPoint(scores)
			else:
				self.callBackEarlyStopping(lossItem)#eye does not use this line
			# 早停
			if self.stop_training==True:
				print('Stop Training!!!')
				break
			time_epoch = time.time() - time_stamp
			print('{:03}* {:.2f} mins, left {:.2f} hours to run'.format(i, time_epoch/60, time_epoch/60/60*(epochs-i)))
		self.desc()
		print(self.bests)
		logTime = '\nRunning {:.2f} hours for {} epochs!'.format((time.time() - time_fit_begin)/60/60, epochs)
		self.logTxt.append(logTime)
		self.logTxt.append(str(self.bests))
		with open(self.root + '/logs.txt', 'w') as f:
			f.write('\n'.join(self.logTxt))
		if hasattr(self.model, 'tmp'):
			tensorboard_logs(self.model.tmp, root=self.root)

	losSCA = 0
	def train(self):
		torch.set_grad_enabled(True)
		self.model.train()  
		lossItem = 0
		tbar = tqdm.tqdm(self.dataset.trainSet(bs=self.args.bs))

		losLoop = 0
		self.losSCA = 0
		for i, imgs in enumerate(tbar):
			losStr = ''
			costList = []
			losItem = 0

			(img, lab, fov, aux) = self.dataset.parse(imgs)#cpu
			img = img.to(self.device)
			lab = lab.to(self.device)
			fov = fov.to(self.device)
			# aux = aux.to(self.device)

			out = self.model(img)
			losSEG = self.gradUtil.backward_seg(out, lab, fov, self.model)
			costList.append(losSEG)
			losItem += losSEG.item()
			losStr += ',seg={:.4f}'.format(losSEG.item())

			# if 'mfgu' in self.model.__name__ or 'mfu' in self.model.__name__:
			if self.args.rec and 'mfau' not in self.model.__name__:
				losTen = self.model.encoder.regular_rec()*self.args.coff_rec
				costList.append(losTen)
				losItem += losTen.item()
				losStr += ',rec={:.4f}'.format(losTen.item())
			if self.args.dmf:
				losDMF = self.model.encoder.fcn.regular_dmf(lab)*self.args.coff_dmf
				costList.append(losDMF)
				losItem += losDMF.item()
				losStr += ',dmf={:.4f}'.format(losDMF.item())

			if isinstance(out, (tuple, list)):
				out = out[0]

			# if self.args.con!='':
			# 	los = self.model.regular(lab=lab, fov=fov) * self.args.coff_con
			# 	costList.append(los)
			# 	losStr += '+{}={:.4f}'.format(self.args.con, los.item())
			# if self.args.att:
			# 	losDMF =  self.model.encoder.regular_att()* self.args.coff_att
			# 	costList.append(losDMF)
			# 	losStr += ',att={:.4f}'.format(losDMF.item())

			# #	rotation constraint
			if self.args.oth or self.args.std or self.args.tvl:
				losDict = self.model.encoder.fcn.regular(oth=self.args.oth, std=self.args.std)

				if self.args.tvl:
					losDMF =  losDict['tvl']* self.args.coff_tvl
					costList.append(losDMF)
					losStr += ',tvl={:.4f}'.format(losDMF.item())
				
			losAll = sum(costList)
			self.gradUtil.optimizer.zero_grad()
			losAll.backward()
			lossItem += losAll.item()
			losLoop += lossItem

			self.gradUtil.optimizer.step()
			self.gradUtil.optimizer.zero_grad()

			self.logTxt.append(losStr)
			tbar.set_description('{:03}$ {:.3f}={}'.format(i, lossItem, losStr[1:]))
		return losLoop

	def predict(self, img, *args):
		self.model.eval()
		torch.set_grad_enabled(False)
		# with torch.no_grad():  
		if not (isinstance(img, dict) or isinstance(img, list)):
			img = img.to(self.device)      
		pred = self.model(img)#*fov.to(self.device)
		if isinstance(pred, dict):
			pred = pred['pred']
		if isinstance(pred, (list, tuple)):
			pred = pred[0]
		pred = pred.detach()
		# pred = pred*fov if fov is not None else pred
		return pred.clamp(0, 1)

	def val(self):
		torch.set_grad_enabled(False)
		self.model.eval()     
		self.gradUtil.criterion.weight=1   
		sum_auc = 0
		sum_iou = 0
		sum_f1s = 0
		sum_los = 0
		dataloader = self.dataset.valSet()
		for i, imgs in enumerate(dataloader):
			(img, lab, fov, aux) = self.dataset.parse(imgs) 
			pred = self.predict(img)
			losSum = self.gradUtil.backward_seg(pred, lab.to(self.device), fov.to(self.device), self.model)
			los = losSum.item()
			sum_los += los
			true = lab.squeeze().numpy().astype(np.float32)
			pred = pred.cpu().squeeze().numpy().astype(np.float32)

			true = true.reshape(-1)
			pred = pred.reshape(-1)
			if fov is not None:
				fov = fov.cpu().view(-1).numpy().astype(np.bool)  
				true, pred = true[fov], pred[fov]
				
			true = np.round(true)
			auc = metrics.roc_auc_score(true, pred)
			sum_auc += auc

			pred = np.round(np.clip(pred, 1e-6, 1-1e-6))
			iou = metrics.jaccard_score(true, pred)
			sum_iou += iou
			f1s = metrics.f1_score(true, pred, average='binary')
			sum_f1s += f1s
			print('\r{:03}$ auc={:.4f} & iou={:.4f} & f1s={:.4f}'.format(i, auc, iou, f1s), end='')
		num = len(dataloader)#i+1#
		los = sum_los/num
		scores = {'auc':sum_auc/num, 'iou':sum_iou/num, 'f1s':sum_f1s/num}
		return scores, los
