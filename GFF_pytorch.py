"""
  将gff转换为pytorch版本的代码
  未完成，tensorflow slim版本的代码有很多的问题，

"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU
import torch.utils.data

def Upsampling(inputs, feature_map_shape):
	return nn.Upsample(size = feature_map_shape, scale_factor=None, mode = 'bilinear');

class ConvBlock(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, rate=1):
		super(ConvBlock, self).__init__()
		padding = kernel_size//2
		self.conv = nn.Sequential(
			BatchNorm2d(in_ch),
			ReLU(inplace=True),
			Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation = rate, bias=True)
			)
	def forward(self, x):
		x = self.conv(x)
		return x

class FuseGFFBlock(nn.Module):
	def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
		super(FuseGFFBlock, self).__init__()
		padding = kernel_size//2
		self.conv = nn.Sequential(
			Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
			BatchNorm2d(out_ch),
			ReLU(inplace=True)
			)

class FuseGFFConvBlock(nn.Module):
	def __init__(self):
		pass
	def forward(self, x):
		pass

class ResNetBlock_1(nn.Module):
	def __init__(self, in_ch, out_ch1, out_ch2):
		super(ResNetBlock_1, self).__init__()
		self.conv1 = ConvBlock(in_ch, out_ch1, kernel_size=1)
		self.conv2 = ConvBlock(out_ch1, out_ch1, kernel_size=3)
		self.conv3 = ConvBlock(out_ch1, out_ch2, kernel_size=1)
	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x+residual
		return x

class ResNetBlock_2(nn.Module):
	def __init__(self, in_ch, out_ch1, out_ch2, s = 1, d = 1):
		super(ResNetBlock_2, self).__init__()
		self.conv1 = ConvBlock(in_ch, out_ch1, kernel_size=1, stride=s)
		self.conv2 = ConvBlock(out_ch1, out_ch1, kernel_size=3)
		self.conv3 = ConvBlock(out_ch1, out_ch2, kernel_size=1)
		### something different here
		self.convDown = ConvBlock(in_ch, out_ch2, kernel_size=1, stride=s, rate=d)
	def forward(self, x):
		change = self.conv1(x)
		change = self.conv2(change)
		change = self.conv3(change)
		x = self.convDown(x)
		return x+change

class ConvUpscaleBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, dropout=0.5):
        super(ConvUpscaleBlock, self).__init__()
        self.deconv = nn.Sequential(
        	BatchNorm2d(in_ch),
			ReLU(inplace=True),
			nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1)
        	)
        #self.d1 = nn.Dropout(dropout)

    def forward(self, x):
    	x = self.deconv(x)
        return x

#class InterpBlock(nn.Module):
#	def __init__(self, ):
#		pass
#	def forawrd(self, x):
#		pass

#pyramid pooling module
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class GFFNet(nn.Module):
	def __init__(self, in_ch):
		super(GFFNet, self).__init__()
		self.conv1 = ConvBlock(in_ch, out_ch = 64, kernel_size = 3)
		self.conv2 = ConvBlock(in_ch = 64, out_ch = 64, kernel_size=7, stride=2)# qes ??2?
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = nn.Sequential(
			ResNetBlock_2(in_ch = 64, out_ch1 = 64, out_ch2 = 256, s = 1),
			ResNetBlock_1(in_ch = 256, out_ch1 = 64, out_ch2 = 256),
			ResNetBlock_1(in_ch = 256, out_ch1 = 64, out_ch2 = 256)
			)
		self.gate1 = nn.Sequential(
			Conv2d(256, 256, kernel_size=1, stride=2),
			Conv2d(256, 256, kernel_size=1), 
			nn.Sigmoid()
			)
		
		self.layer2 = nn.Sequential(
			ResNetBlock_2(in_ch = 256, out_ch1 = 128, out_ch2 = 512, s = 2),
			ResNetBlock_1(in_ch = 512, out_ch1 = 128, out_ch2 = 512),
			ResNetBlock_1(in_ch = 512, out_ch1 = 128, out_ch2 = 512),
			ResNetBlock_1(in_ch = 512, out_ch1 = 128, out_ch2 = 512)
			)
		self.gate2 = nn.Sequential(
			Conv2d(512, 256, kernel_size=1),
			Conv2d(256, 256, kernel_size=1), 
			nn.Sigmoid()
			)
		
		self.layer3 = nn.Sequential(
			ResNetBlock_2(in_ch = 512, out_ch1 = 256, out_ch2 = 1024, d = 2),
			ResNetBlock_1(in_ch = 1024, out_ch1 = 256, out_ch2 = 1024),
			ResNetBlock_1(in_ch = 1024, out_ch1 = 256, out_ch2 = 1024),
			ResNetBlock_1(in_ch = 1024, out_ch1 = 256, out_ch2 = 1024),
			ResNetBlock_1(in_ch = 1024, out_ch1 = 256, out_ch2 = 1024),
			ResNetBlock_1(in_ch = 1024, out_ch1 = 256, out_ch2 = 1024)
			)
		self.gate3 = nn.Sequential(
			Conv2d(1024, 256, kernel_size=1),
			Conv2d(256, 256, kernel_size=1), 
			nn.Sigmoid()
			)

		self.layer4 = nn.Sequential(
			ResNetBlock_2(in_ch = 1024, out_ch1 = 512, out_ch2 = 2048, d = 4),
			ResNetBlock_1(in_ch = 2048, out_ch1 = 512, out_ch2 = 2048),
			ResNetBlock_1(in_ch = 2048, out_ch1 = 512, out_ch2 = 2048)
			)
		self.gate4 = nn.Sequential(
			Conv2d(2048, 256, kernel_size=1),
			Conv2d(256, 256, kernel_size=1),
			nn.Sigmoid()
			)
	def forward(self, x):


"""
  GFFNET构成:
   conv-3*3
   conv-7*7
   pool
   10 * conv-3*3    x1  x1n
   13 * conv-3*3    x2  
   19 * conv-3*3    x3
   10 * conv-3*3    x4
"""