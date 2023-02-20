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

from data import *
from nets import *
from build import *
from utils import *

from loop import *

import argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="Train network")
#	实验参数
parser.add_argument('--inc', type=str, default='', help='instruction')
parser.add_argument('--gpu', type=int, default=0, help='cuda number')
parser.add_argument('--los', type=str, default='fr', help='loss function')
parser.add_argument('--net', type=str, default='mfu', help='network')
parser.add_argument('--seg', type=str, default='gauss', help='network')
parser.add_argument('--db', type=str, default='drive', help='instruction')
parser.add_argument('--loo', type=int, default=99, help='Leave One Out')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--ds', type=int, default=128, help='batch size')
parser.add_argument('--pl', type=str2bool, default=False, help='Parallel!')
parser.add_argument('--root', type=str, default='', help='root folder')
parser.add_argument('--coff_ds', type=float, default=0.1, help='Cofficient of Deep Supervision!')

#	数据增强
parser.add_argument('--sca', type=str, default='', choices=['scs','scd',''], help='spurious correlation surface or domain!')
parser.add_argument('--coff_sca', type=float, default=0.3, help='Cofficient of sca!')
parser.add_argument('--coff_tv', type=float, default=100, help='Cofficient of TVLoss!')
parser.add_argument('--type_tv', type=int, default=2, help='Cofficient of TVLoss!')
parser.add_argument('--grid_rsf', type=int, default=4, help='Cofficient of TVLoss!')

parser.add_argument('--oth', type=str2bool, default=False, help='rotation match filter!')
parser.add_argument('--coff_oth', type=float, default=1, help='Cofficient of regualar rotation!')
parser.add_argument('--std', type=str2bool, default=False, help='rotation match filter!')
parser.add_argument('--coff_std', type=float, default=1, help='Cofficient of regualar rotation!')
parser.add_argument('--att', type=str2bool, default=False, help='rotation match filter!')
parser.add_argument('--coff_att', type=float, default=.1, help='Cofficient of regualar rotation!')
parser.add_argument('--tvl', type=str2bool, default=False, help='rotation match filter!')
parser.add_argument('--coff_tvl', type=float, default=10, help='Cofficient of regualar rotation!')

parser.add_argument('--dmf', type=str2bool, default=True, help='rotation match filter!')
parser.add_argument('--coff_dmf', type=float, default=10, help='Cofficient of regualar rotation!')
parser.add_argument('--rec', type=str2bool, default=True, help='reconstruct attention map!')
parser.add_argument('--coff_rec', type=float, default=0.1, help='Cofficient of reconstruct attention map!')

#	对比学习相关参数
parser.add_argument('--con', type=str, default='', choices=['','corr','cos','wht'], help='Whitening or Contrastive Learning!')
parser.add_argument('--num_smp', type=int, default=48, help='sampler number')
parser.add_argument('--num_con', type=int, default=32, help='contrastive number')
parser.add_argument('--coff_con', type=float, default=0.1, help='Cofficient of Contrastive learning!')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 训练程序########################################################
if __name__ == '__main__':
	dataset = EyeSetGenerator(dbname=args.db, loo=args.loo, datasize=args.ds) 

	expStr = ''
	if args.db=='stare' and args.loo>=0 and args.loo<20:
		expStr += 'LOO'+str(args.loo)

	net = build_model(args.net, args.seg, args)
	net.encoder.flag_scale = True if dataset.dbname in ['chase', 'hrf'] else False
	print('@'*32, 'flag_scale=', net.encoder.flag_scale, dataset.dbname)
	if args.con!='':
		net.con.card_select = args.num_smp
		expStr += args.con.upper()+str(args.num_smp) +'_'+str(args.coff_con)

	keras = KerasTorch(model=net, args=args) 
	keras.args = args
	
	if args.sca!='':
		expStr += f'{args.sca.upper()}{args.grid_rsf}_{args.coff_sca}'
		
	# args.rec = 'mfgu' in net.__name__ or 'mfu' in net.__name__
	if args.rec:
		expStr += '_REC'+str(args.coff_rec)
	if args.dmf:
		expStr += '_DMF'+str(args.coff_dmf)
	if args.seg=='gauss':
		# if args.rot:
		# 	expStr += '_ROT'+str(args.coff_rot)
		if args.oth:
			expStr += '_OTH'+str(args.coff_oth)
		if args.std:
			expStr += '_STD'+str(args.coff_std)
		if args.tvl:
			expStr += '_TVL'+str(args.coff_tvl)
		if args.att:
			expStr += '_Att'+str(args.coff_att)

	net.__name__ += expStr + 'bs'+str(args.bs)+ 'ds'+str(args.coff_ds) + args.inc

	print('Network Name:', net.__name__)
	keras.compile(dataset, loss=args.los, lr=0.01)  
	keras.gradUtil.coff_ds = args.coff_ds
	keras.fit(epochs=121)   
