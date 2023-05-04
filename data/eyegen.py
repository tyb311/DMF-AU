# -*- coding: utf-8 -*-
import sys
sys.path.append('.')
sys.path.append('..')

if __name__ == '__main__':
	from trans import *
	from eyenpy import *
else:
	try:
		from .trans import *
		from .eyenpy import *
	except:
		from data.trans import *
		from data.eyenpy import *

#start#
import imgaug as ia
import imgaug.augmenters as iaa
IAA_NOISE = iaa.OneOf(children=[# Noise
		iaa.Add((-7, 7), per_channel=True),
		iaa.AddElementwise((-7, 7)),
		iaa.Multiply((0.9, 1.1), per_channel=True),
		iaa.MultiplyElementwise((0.9, 1.1), per_channel=True),

		iaa.AdditiveGaussianNoise(scale=3, per_channel=True),
		iaa.AdditiveLaplaceNoise(scale=3, per_channel=True),
		iaa.AdditivePoissonNoise(lam=5, per_channel=True),

		iaa.SaltAndPepper(0.01, per_channel=True),
		iaa.ImpulseNoise(0.01),
	]
)
TRANS_NOISE = IAA_NOISE

from albumentations import (
	Flip, Transpose, RandomRotate90, PadIfNeeded, RandomGridShuffle,
	OneOf, Compose, CropNonEmptyMaskIfExists, CLAHE, RandomGamma
) 

TRANS_TEST = Compose([CLAHE(p=1), RandomGamma(p=1)])#
TRANS_FLIP = Compose([
	OneOf([Transpose(p=1), RandomRotate90(p=1), ], p=.7),Flip(p=.7), 
])


from skimage import filters
from torch.utils.data import DataLoader, Dataset
class EyeSetGenerator(Dataset, EyeSetResource):
	exeNums = {'train':Aug4CSA.number, 'val':Aug4Val.number, 'test':1}
	exeMode = 'train'#train, val, test
	exeData = 'train'#train, test, full

	SIZE_IMAGE = 384
	expCross = False   
	LEN_AUG = 32
	radius = {'drive':7.6, 'stare':9.35, 'chase':13.0, 'hrf':13.9}		#morph_cross
	def __init__(self, datasize=128, **args):
		super(EyeSetGenerator, self).__init__(**args)
		self.SIZE_IMAGE = datasize
		self.LEN_AUG = 96 // (datasize//64)**2
		print('SIZE_IMAGE:{} & AUG SIZE:{}'.format(self.SIZE_IMAGE, self.LEN_AUG))

		
	def __len__(self):
		length = self.lens[self.exeData]*self.exeNums[self.exeMode]
		if self.isTrainMode:
			if self.dbname=='hrf':
				return length*8
			return length*self.LEN_AUG
		return length

	def set_mode(self, mode='train'):
		self.exeMode = mode
		self.exeData = 'full' if self.expCross else mode 
		self.isTrainMode = (mode=='train')
		self.isValMode = (mode=='val')
		self.isTestMode = (mode=='test')
	def trainSet(self, bs=8, data='train'):#pin_memory=True, , shuffle=True
		self.set_mode(mode='train')
		return DataLoader(self, batch_size=bs, pin_memory=True, num_workers=4, shuffle=True)
	def valSet(self, data='val'):
		self.set_mode(mode='val')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	def testSet(self, data='test'):
		self.set_mode(mode='test')
		return DataLoader(self, batch_size=1,  pin_memory=True, num_workers=2)
	#DataLoader worker (pid(s) 5220) exited unexpectedly, 令numworkers>1就可以啦


	# @staticmethod
	def parse(self, pics, cat=True):
		rows, cols = pics['lab'].squeeze().shape[-2:]     
		for key in pics.keys(): 
			# print(key, pics[key].shape)
			pics[key] = pics[key].view(-1,1,rows,cols) 

		return pics['img'], torch.round(pics['lab']), torch.round(pics['fov']), pics['aux']

	def post(self, img, lab, fov):
		if type(img) is not np.ndarray:img = img.squeeze().cpu().numpy()
		if type(lab) is not np.ndarray:lab = lab.squeeze().cpu().numpy()
		if type(fov) is not np.ndarray:fov = fov.squeeze().cpu().numpy()
		img = img * fov
		return img, lab, fov

	use_csm = False
	mfg_cat = False
	def __getitem__(self, idx, divide=32):
		index = idx % self.lens[self.exeData] 
		pics = self.readDict(index, self.exeData)
		imag = pics['img']# = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)

		# pics['aux'] = pics['fov']
		if self.isTrainMode:
			# print(pics['lab'].shape, pics['fov'].shape, pics['aux'].shape)
			mask = np.stack([pics['lab'], pics['fov'], pics['aux']], axis=-1)

			# 裁剪增强
			augCrop = CropNonEmptyMaskIfExists(p=1, height=self.SIZE_IMAGE, width=self.SIZE_IMAGE)
			picaug = augCrop(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']

			imag = TRANS_TEST(image=imag)['image']

			# 添加噪声
			imag = TRANS_NOISE(image=imag)
			# 变换增强
			picaug = TRANS_FLIP(image=imag, mask=mask)
			imag, mask = picaug['image'], picaug['mask']
			
			pics['img'] = imag
			pics['lab'], pics['fov'], pics['aux'] = mask[:,:,0],mask[:,:,1],mask[:,:,2]

		else:
			pics['img'] = TRANS_TEST(image=pics['img'])['image']
			# 图像补齐
			h, w = pics['lab'].shape
			w = int(np.ceil(w / divide)) * divide
			h = int(np.ceil(h / divide)) * divide
			augPad = PadIfNeeded(p=1, min_height=h, min_width=w)
			for key in pics.keys():
				pics[key] = augPad(image=pics[key])['image']

			if self.isValMode:# 验证增强->非测试，则增强
				flag = idx//self.lens[self.exeData]
				pics = Aug4Val.forward(pics, flag)

		if pics['img'].shape[-1]==3:#	green or gray
			pics['img'] = cv2.cvtColor(pics['img'], cv2.COLOR_RGB2GRAY)
			# pics['img'] = pics['img'][:,:,1]#莫非灰度图像比绿色通道更好一点？
	
		# pics['aux'] = filters.threshold_local(pics['img'], 25, method='median') #返回一个阈值图像
		for key in pics.keys():
			# print(key, pics[key].shape)
			pics[key] = torch.from_numpy(pics[key]).type(torch.float32).div(255)
		return pics
#end#



def tensor2image(x):
	x = x.squeeze().data.numpy()
	return x



if __name__ == '__main__':
	from eyeimg import *
	udb = EyeSetDomain(ignore='stare').infSet(4)

	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='drive', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='hrf', isBasedPatch=True)#
	# db = EyeSetGenerator(folder='../datasets/seteye', dbname='chase', isBasedPatch=True)#
	db = EyeSetGenerator(folder='../datasets/seteye', dbname='stare', loo=0)#
	# db = EyeSetGenerator(folder=r'G:\Objects\expSeg\datasets\seteye', dbname='drive', isBasedPatch=False)#
	# db.expCross = True
	print('generator:', len(db.trainSet()), len(db.valSet()), len(db.testSet()), )

	# db.mfg_cat = True
	db.use_sca = True
	for i, imgs in enumerate(db.trainSet(4)):
	# for i, imgs in enumerate(db.valSet(1)):
	# for i, imgs in enumerate(db.testSet()):
		# print(imgs.keys())
		# print(imgs)
		(img, lab, fov, aux) = db.parse(imgs)
		print(img.shape, lab.shape, fov.shape, aux.shape)
		# fov = torch_dilation(lab)


		# _,_,h,w = img.shape
		# stW,stH = random.randint(1,w),random.randint(1,h)
		# matts = torch.ones_like(lab, device=lab.device)
		# matts[:,:,:stH,:stW] = 0
		# matts[:,:,stH:,stW:] = 0
		
		roi = img[fov>0]
		print('Range of Style:', roi.min().item(), roi.max().item())

		if img.shape[0]!=1:
			img = img[0]
			lab = lab[0]
			fov = fov[0]
			aux = aux[0]
		

		plt.subplot(221),plt.imshow(tensor2image(img))
		plt.subplot(222),plt.imshow(tensor2image(lab))
		plt.subplot(223),plt.imshow(tensor2image(fov))
		plt.subplot(224),plt.imshow(tensor2image(aux))
		plt.show()