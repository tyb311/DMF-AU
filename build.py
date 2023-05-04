
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
from torchvision.transforms import functional as f
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



import sys
sys.path.append('.')
sys.path.append('..')

from utils import *
from nets import *
from nets.siam import SIAM



def build_model(type_net='dmf', type_seg='', args=None):
	print('build_model:', type_net, type_seg, args.con)
	model = eval(type_net+'(in_channels=1, num_con=args.num_con, type_seg=type_seg)')
		
	model = SIAM(encoder=model, num_con=args.num_con, con=args.con)
	if type_seg=='lunet':
		model.__name__ = '{}{}'.format(type_net, 'U')
	elif type_net=='dmf':
		model.__name__ = '{}{}'.format('MF', type_seg)
	return model