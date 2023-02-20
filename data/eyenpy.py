# -*- encoding:utf-8 -*-
#start#
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
import socket
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def gain(ret, p=1):    #gain_off
	mean = np.mean(ret)
	ret_min = mean-(mean-np.min(ret))*p
	ret_max = mean+(np.max(ret)-mean)*p
	ret = 255*(ret - ret_min)/(ret_max - ret_min)
	ret = np.clip(ret, 0, 255).astype(np.uint8)
	return ret

def arr2img(pic):
	return Image.fromarray(pic.astype(np.uint8))#, mode='L'

def arrs2imgs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = arr2img(pic[key])
	return _pic

def imgs2arrs(pic):
	_pic=dict()
	for key in pic.keys():
		_pic[key] = np.array(pic[key])
	return _pic

def pil_tran(pic, tran=None):
	if tran is None:
		return pic
	if isinstance(tran, list):
		for t in tran:
			for key in pic.keys():
				pic[key] = pic[key].transpose(t)
	else:
		for key in pic.keys():
			pic[key] = pic[key].transpose(tran)
	return pic

class Aug4Val(object):
	number = 8
	@staticmethod
	def forward(pic, flag):
		flag %= Aug4Val.number
		if flag==0:
			return pic
		pic = arrs2imgs(pic)
		if flag==1:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
		if flag==2:
			return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
		if flag==3:
			return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
		if flag==4:
			return imgs2arrs(pil_tran(pic, tran=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
		if flag==5:
			return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_TOP_BOTTOM]))
		if flag==6:
			return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT]))
		if flag==7:
			return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))


class EyeSetResource(object):
	size = dict()
	save = {
		'drive':{'h':584,'w':565},
		'chase':{'h':960,'w':999},
		'stare':{'h':605,'w':700},
		'hrf':  {'h':1168,'w':1752},
		}
	def __init__(self, folder='../eyeset', dbname='drive', loo=99, desc=True, **args):
		super(EyeSetResource, self).__init__()
		
		self.folder = '../datasets/seteye'
		# else:
		# 	raise EnvironmentError('No thi root!')
		# self.folder = folder
		self.dbname = dbname

		self.imgs, self.labs, self.fovs, self.auxs = self.getDataSet(self.dbname)
		if dbname=='stare' and loo>=0 and loo<20: 
			self.imgs['test'] = [self.imgs['full'][loo]]
			self.imgs['train'] = self.imgs['full'][:loo] + self.imgs['full'][1+loo:]
			self.imgs['val'] = self.imgs['train']
			
			self.labs['test'] = [self.labs['full'][loo]]
			self.labs['train'] = self.labs['full'][:loo] + self.labs['full'][1+loo:]
			self.labs['val'] = self.labs['train']
			
			self.fovs['test'] = [self.fovs['full'][loo]]
			self.fovs['train'] = self.fovs['full'][:loo] + self.fovs['full'][1+loo:]
			self.fovs['val'] = self.fovs['train']
			
			self.auxs['test'] = [self.auxs['full'][loo]]
			self.auxs['train'] = self.auxs['full'][:loo] + self.auxs['full'][1+loo:]
			self.auxs['val'] = self.auxs['train']

			print('LOO:', loo, self.imgs['test'])
			print('LOO:', loo, self.labs['test'])
			print('LOO:', loo, self.fovs['test'])
			print('LOO:', loo, self.auxs['test'])

		self.lens = {'train':len(self.labs['train']),   'val':len(self.labs['val']),
					 'test':len(self.labs['test']),     'full':len(self.labs['full'])}  
		# print(self.lens)  
		if self.lens['test']>0:
			lab = self.readArr(self.labs['test'][0])
			self.size['raw'] = lab.shape
			h,w = lab.shape
			self.size['pad'] = (math.ceil(h/32)*32, math.ceil(w/32)*32)
			# print('size:', self.size)
		else:
			print('dataset has no images!')

		if desc:
			# print('*'*32,'eyeset','*'*32)
			strNum = 'images:{}+{}+{}#{}'.format(self.lens['train'], self.lens['val'], self.lens['test'], self.lens['full'])
			print('{}@{}'.format(self.dbname, strNum))

	def getDataSet(self, dbname):        
		# 测试集
		imgs_test = self.readFolder(dbname, part='test', image='rgb')
		labs_test = self.readFolder(dbname, part='test', image='lab')
		fovs_test = self.readFolder(dbname, part='test', image='fov')
		auxs_test = self.readFolder(dbname, part='test', image='aux')
		# 训练集
		imgs_train = self.readFolder(dbname, part='train', image='rgb')
		labs_train = self.readFolder(dbname, part='train', image='lab')
		fovs_train = self.readFolder(dbname, part='train', image='fov')
		auxs_train = self.readFolder(dbname, part='train', image='aux')
		# 全集
		imgs_full,labs_full,fovs_full,auxs_full = [],[],[],[]
		imgs_full.extend(imgs_train); imgs_full.extend(imgs_test)
		labs_full.extend(labs_train); labs_full.extend(labs_test)
		fovs_full.extend(fovs_train); fovs_full.extend(fovs_test)
		auxs_full.extend(auxs_train); auxs_full.extend(auxs_test)

		db_imgs = {'train': imgs_train, 'val':imgs_train, 'test': imgs_test, 'full':imgs_full}
		db_labs = {'train': labs_train, 'val':labs_train, 'test': labs_test, 'full':labs_full}
		db_fovs = {'train': fovs_train, 'val':fovs_train, 'test': fovs_test, 'full':fovs_full}
		db_auxs = {'train': auxs_train, 'val':auxs_train, 'test': auxs_test, 'full':auxs_full}
		return db_imgs, db_labs, db_fovs, db_auxs

	def readFolder(self, dbname, part='train', image='rgb'):
		path = self.folder + '/' + dbname + '/' + part + '_' + image
		imgs = glob.glob(path + '/*.npy')
		imgs.sort()
		return imgs
		
	def readArr(self, image):
		# assert(image.endswith('.npy'), 'not npy file!') 
		return np.load(image) 
	
	def readDict(self, index, exeData):  
		img = self.readArr(self.imgs[exeData][index])
		fov = self.readArr(self.fovs[exeData][index])
		lab = self.readArr(self.labs[exeData][index])
		aux = fov#self.readArr(self.auxs[exeData][index])
		if fov.shape[-1]==3:
			fov = cv2.cvtColor(fov, cv2.COLOR_BGR2GRAY)
		return {'img':img, 'lab':lab, 'fov':fov, 'aux':aux}
#end#


'''
max-width of retinal datasets
STARE:	5.4
DRIVE:	6.9
CHASE:	9.25
HRF:	20.06/2=10.03
'''


if __name__ == '__main__':
	# main()
	# crop4trainset()		

	db = EyeSetResource(folder='G:\Objects\datasets\seteye', dbname='drive')
	# # db = EyeSetResource(folder='../eyeset', dbname='stare')

	# dataset2npy(db)
	mode = 'val'
	for i in range(db.lens[mode]):
		pics = db.readDict(i, mode)
		a,b,c,d = pics['img'],pics['lab'],pics['fov'],pics['aux']
		plt.subplot(221),plt.imshow(a)
		plt.subplot(222),plt.imshow(b)
		plt.subplot(223),plt.imshow(c)
		plt.subplot(224),plt.imshow(d)
		plt.show()

