# -*- encoding:utf-8 -*-
# 常用资源库
import math,cv2,random
from PIL import Image
# 图像显示
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import numpy as np
import onnxruntime as ort

from albumentations import (
	PadIfNeeded, CLAHE, RandomGamma, CenterCrop, Compose
) # 图像变换函数

TRANS_TEST = Compose([CLAHE(p=1), RandomGamma(p=1)])#




class NetWork:
	def __init__(self, 
		onnx_vessel=r"onnx/dmfau_drive.onnx"
		):
		self.session_vessel = ort.InferenceSession(onnx_vessel)
		print(self.session_vessel.__dict__)
		self.session_vessel.scale_factor = 1

	tmp={}
	def forward(self, img, show=False, save=False, resize=True, crop=False, inc=''):
		# 按照比例对图像进行缩放，以适应血管最大宽度
		if resize:
			h,w = img.shape[:2]
			if h>3000:
				ratio = 0.125
			elif h>2000:
				ratio = 0.25
			elif h>1000:
				ratio = 0.5
			else:
				ratio = 1
			# print(img.shape, img.dtype, (int(w*ratio),int(h*ratio)))
			img = cv2.resize(img, dsize=(int(w*ratio),int(h*ratio)))

		h,w = img.shape[:2]
		print('shape-resize:', h, w)
		augCrop = CenterCrop(h,w)

		# 对图像进行裁剪为32的倍数
		pad = (math.ceil(h/32)*32, math.ceil(w/32)*32)
		h, w = pad
		augPadd = PadIfNeeded(p=1, min_height=h, min_width=w)
		print('shape-padding:', h, w)

		img = augPadd(image=img)['image']
		img = TRANS_TEST(image=img)['image']

		raw = img
		img = img[:,:,1]
		h,w = img.shape
		img = img.reshape(1,1,h,w).astype(np.float32)/255

		# 预测血管和视杯视盘
		out_vessel = self.session_vessel.run(None, {"input": img}, )[0].squeeze()
		out = (out_vessel>0.5).astype(np.uint8)*255
		print('shape-output:', out.shape)

		# 保存或显示
		if save:
			Image.fromarray(raw).save(r'source_image{}.png'.format(inc))
			Image.fromarray(out).save(r'segmentation{}.png'.format(inc))
		if show:
			plt.subplot(121),plt.imshow(raw)
			plt.subplot(122),plt.imshow(out)
			plt.show()

		if crop:
			out = augCrop(image=out)['image']
		return out







if __name__ == '__main__':

	img=r'source_image.png'
	img = np.array(Image.open(img))

	net = NetWork("dmfau_stare.onnx")
	net.forward(img, show=True, save=True)

