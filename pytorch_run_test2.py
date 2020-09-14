from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net, Resnet_Unet
from Models2 import reS_Unet
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time



train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Testing on CPU')
else:
    print('CUDA is available. Testing on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 8
print('batch_size = ' + str(batch_size))
#valid_size = 0.15
epoch = 50
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

#plotter = VisdomLinePlotter(env_name='Tutorial Plots')

#######################################################
#Setting up the model
#######################################################

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Resnet_Unet, reS_Unet]

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

#passsing this string so that if it's AttU_Net or R2ATTU_Net it doesn't throw an error at torchSummary


model_test = model_unet(model_Inputs[0], 3, 1)

model_test.to(device)

#######################################################
#Getting the Summary of Model
#######################################################

#torchsummary.summary(model_test, input_size=(3, 128, 128))

#######################################################
#Passing the Dataset of Images and Labels
#######################################################

t_data = '/kaggle/input/isic2017/ISIC2017_train_imgs/'
l_data = '/kaggle/input/isic2017/ISIC2017_train_labels/'
test_image = '/kaggle/input/isic2017/ISIC2017_test_imgs/ISIC_0012086.jpg'
test_label = '/kaggle/input/isic2017/ISIC2017_test_labels/ISIC_0012086_segmentation.png'
test_folderP = '/kaggle/input/isic2017/ISIC2017_test_imgs/*'
test_folderL = '/kaggle/input/isic2017/ISIC2017_test_labels/*'


data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
data_transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

#######################################################
#Loading the model
#######################################################

#test1 =model_test.load_state_dict(torch.load('./model/Unet_D_' +
#                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
#                   + '_batchsize_' + str(batch_size) + '.pth'))

#######################################################
#checking if cuda is available
#######################################################

if torch.cuda.is_available():
    torch.cuda.empty_cache()

#######################################################
#Loading the model
#######################################################

model_test.load_state_dict(torch.load('/kaggle/input/trainedmodel/Unet_epoch_50_batchsize_8.pth'))

model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort


read_test_folder112 = './model/gen_images'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './model/pred_threshold'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './model/label_threshold'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)




#######################################################
#saving the images in the files
#######################################################

img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i])

    #im1 = im
    #im_n = np.array(im1)
    #im_n_flat = im_n.reshape(-1, 1)

    #for j in range(im_n_flat.shape[0]):
    #    if im_n_flat[j] != 0:
    #        im_n_flat[j] = 255

    s = data_transform(im)
    pred = model_test(s.unsqueeze(0).cuda()).cpu()
    pred = F.sigmoid(pred)
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    f_name = x_sort_test[i].split('.')
    f_name = f_name[0].split('/')
    f_name = f_name[-1]
    #original saving method
    #x1 = plt.imsave('./model/gen_images/im_epoch_' + str(epoch) + 'int_' + str(i)
    #                + '_img_no_' + str(img_test_no) + '.png', pred[0][0])
    x1 = plt.imsave('./model/gen_images/'+f_name+'_prediction.png', pred[0][0])


####################################################
#Calculating the Dice Score
####################################################

data_transform31 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.Grayscale(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
data_transform32 = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.CenterCrop(96),
            torchvision.transforms.Grayscale(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


read_test_folderP = glob.glob('./model/gen_images/*')
#changes
x_sort_testP = natsort.natsorted(read_test_folderP)


read_test_folderL = glob.glob(test_folderL)
#changes
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


dice_score123 = 0.0
x_count = 0
x_dice = 0### used for selection
t_acc, t_sen, t_spe, t_dice, t_jacc = 0.0,0.0,0.0,0.0,0.0

for i in range(len(read_test_folderP)):

    x = Image.open(x_sort_testP[i])
    s = data_transform31(x)
    s = np.array(s)
    s = threshold_predictions_v(s)

    #save the images
    x1 = plt.imsave('./model/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i])
    s2 = data_transform32(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)

    #save the Images
    y1 = plt.imsave('./model/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)

    total = dice_coeff(s, s3)
    acc, sen, spe, dice, jacc = accuracy_score(s, s3)
    names = x_sort_testP[i].split('/')
    print(names[-1]+'(dice1): '+str(total))
    print('accuracy:%.5f, sensitivity:%.5f, specificity:%.5f, dice:%.5f, jaccard:%.5f'%(acc, sen, spe, dice, jacc))
    t_acc += acc
    t_sen += sen
    t_spe += spe
    t_dice += dice
    t_jacc += jacc
    if total <= 0.3:
        x_count += 1
    if total > 0.3:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total


print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))
print('accuracy:%.5f, sensitivity:%.5f, specificity:%.5f, dice:%.5f, jaccard:%.5f'%( t_acc/len(read_test_folderP), 
	t_sen/len(read_test_folderP), t_spe/len(read_test_folderP), t_dice/len(read_test_folderP), t_jacc/len(read_test_folderP)))#print(x_count)
#print(x_dice)
#print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))

