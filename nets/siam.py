# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets import *

class SIAM(nn.Module):
	__name__ = 'siam'
	def __init__(self,
				 encoder,
				 num_con=64,
				 con='wht',
				 **kwargs):
		super().__init__()
		self.con = SupCon(dim=num_con, con=con)

		self.encoder = encoder
		# self.__name__ = self.encoder.__name__
		self.__name__ = self.encoder.__name__
		
		self.projector = self.encoder.projector
		self.predictor = self.encoder.predictor

	train_geo=False
	def forward(self, img, **args):#只需保证两组数据提取出的特征标签顺序一致即可
		out = self.encoder(img, **args)
		self.pred = self.encoder.pred
		self.feat = self.encoder.feat
		# self._dequeue_and_enqueue(proj1_ng, proj2_ng)
		if hasattr(self.encoder, 'tmp'):
			self.tmp = self.encoder.tmp
		return out

	def constraint(self, **args):
		return self.encoder.constraint(**args)

	def regular(self, lab, fov=None):#contrastive loss split by classification
		feat = self.con.select2(self.feat.clone(), self.pred.detach(), lab, fov)
		# print(emb.shape)
		proj = self.projector(feat)
		pred = self.predictor(proj)
		# print(proj.shape, feat.shape)
		fhj, flj, bhj, blj = torch.chunk(proj, 4)
		fhd, fld, bhd, bld = torch.chunk(pred, 4)
		# return 0
		losPos = self.con(fhj.detach(), fld) + self.con(fhd.detach(), flj)
		losNeg = self.con(bhj.detach(), bld) + self.con(bhd.detach(), blj)
		los = losPos + losNeg
		return los