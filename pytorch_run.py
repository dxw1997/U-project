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
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
from utils import LR_Scheduler
import time
#from ploting import VisdomLinePlotter
#from visdom import Visdom


#######################################################
#Checking if GPU is used
#######################################################

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#######################################################
#Setting the basic paramters of the model
#######################################################

batch_size = 8
print('batch_size = ' + str(batch_size))

valid_size = 0.15

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

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet, Resnet_Unet]


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

t_data = '/home/dxw/Dataset/ISIC2017_train_imgs/'
l_data = '/home/dxw/Dataset/ISIC2017_train_labels/'
test_image = '/home/dxw/Dataset/ISIC2017_test_imgs/ISIC_0012086.jpg'
test_label = '/home/dxw/Dataset/ISIC2017_test_labels/ISIC_0012086_segmentation.png'
test_folderP = '/home/dxw/Dataset/ISIC2017_test_imgs/*'
test_folderL = '/home/dxw/Dataset/ISIC2017_test_labels/*'

Training_Data = Images_Dataset_folder(t_data,
                                      l_data)

#######################################################
#Giving a transformation for input data
#######################################################

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
#Trainging Validation Split
#######################################################

num_train = len(Training_Data)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=train_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=batch_size, sampler=valid_sampler,
                                           num_workers=num_workers, pin_memory=pin_memory,)

#######################################################
#Using Adam as Optimizer
#######################################################

initial_lr = 0.001
opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr) # try SGD
#opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)

MAX_STEP = int(500)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-8)
#scheduler = optim.lr_scheduler.CosineAnnealingLr(opt, epoch, 1)
scheduler = LR_Scheduler('cos', 0.001, 50, int(2000/8))

#######################################################
#Writing the params to tensorboard
#######################################################

#writer1 = SummaryWriter()
#dummy_inp = torch.randn(1, 3, 128, 128)
#model_test.to('cpu')
#writer1.add_graph(model_test, model_test(torch.randn(3, 3, 128, 128, requires_grad=True)))
#model_test.to(device)

#######################################################
#Creating a Folder for every data of the program
#######################################################

New_folder = './model'

if os.path.exists(New_folder) and os.path.isdir(New_folder):
    shutil.rmtree(New_folder)

try:
    os.mkdir(New_folder)
except OSError:
    print("Creation of the main directory '%s' failed " % New_folder)
else:
    print("Successfully created the main directory '%s' " % New_folder)

#######################################################
#Setting the folder of saving the predictions
#######################################################

read_pred = './model/pred'

#######################################################
#Checking if prediction folder exixts
#######################################################

if os.path.exists(read_pred) and os.path.isdir(read_pred):
    shutil.rmtree(read_pred)

try:
    os.mkdir(read_pred)
except OSError:
    print("Creation of the prediction directory '%s' failed of dice loss" % read_pred)
else:
    print("Successfully created the prediction directory '%s' of dice loss" % read_pred)

#######################################################
#checking if the model exists and if true then delete
#######################################################

read_model_path = './model/Unet_D_' + str(epoch) + '_' + str(batch_size)

if os.path.exists(read_model_path) and os.path.isdir(read_model_path):
    shutil.rmtree(read_model_path)
    print('Model folder there, so deleted for newer one')

try:
    os.mkdir(read_model_path)
except OSError:
    print("Creation of the model directory '%s' failed" % read_model_path)
else:
    print("Successfully created the model directory '%s' " % read_model_path)

#######################################################
#Training loop
#######################################################

for i in range(epoch):

    train_loss = 0.0
    valid_loss = 0.0
    since = time.time()
    #original
    #scheduler.step(i)
    #lr = scheduler.get_lr()

    #######################################################
    #Training Data
    #######################################################

    model_test.train()
    k = 1
    c = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        #If want to get the input images with their Augmentation - To check the data flowing in net
        #input_images(x, y, i, n_iter, k)

       # grid_img = torchvision.utils.make_grid(x)
        #writer1.add_image('images', grid_img, 0)

       # grid_lab = torchvision.utils.make_grid(y)

        opt.zero_grad()

        y_pred = model_test(x)
        lossT = calc_loss(y_pred, y)     # Dice_loss Used

        train_loss += lossT.item() * x.size(0)
        lossT.backward()
      #  plot_grad_flow(model_test.named_parameters(), n_iter)
        opt.step()
        x_size = lossT.item() * x.size(0)
        scheduler(opt, c, i)
        c = c+1
        k = 2

    #    for name, param in model_test.named_parameters():
    #        name = name.replace('.', '/')
    #        writer1.add_histogram(name, param.data.cpu().numpy(), i + 1)
    #        writer1.add_histogram(name + '/grad', param.grad.data.cpu().numpy(), i + 1)


    #######################################################
    #Validation Step
    #######################################################

    model_test.eval()
    torch.no_grad() #to increase the validation process uses less memory

    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)

        y_pred1 = model_test(x1)
        lossL = calc_loss(y_pred1, y1)     # Dice_loss Used

        valid_loss += lossL.item() * x1.size(0)
        x_size1 = lossL.item() * x1.size(0)

    #######################################################
    #Saving the predictions
    #######################################################

    im_tb = Image.open(test_image)
    im_label = Image.open(test_label)
    s_tb = data_transform(im_tb)
    s_label = data_transform2(im_label)
    s_label = s_label.detach().numpy()

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

   #pred_tb = threshold_predictions_v(pred_tb)

    x1 = plt.imsave(
        './model/pred/img_iteration_' + str(n_iter) + '_epoch_'
        + str(i) + '.png', pred_tb[0][0])

  #  accuracy = accuracy_score(pred_tb[0][0], s_label)

    #######################################################
    #To write in Tensorboard
    #######################################################

    train_loss = train_loss / len(train_idx)
    valid_loss = valid_loss / len(valid_idx)

    if (i+1) % 1 == 0:
        print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(i + 1, epoch, train_loss,
                                                                                      valid_loss))
 #       writer1.add_scalar('Train Loss', train_loss, n_iter)
  #      writer1.add_scalar('Validation Loss', valid_loss, n_iter)
        #writer1.add_image('Pred', pred_tb[0]) #try to get output of shape 3


    #######################################################
    #Early Stopping
    #######################################################

    if valid_loss <= valid_loss_min and epoch_valid >= i: # and i_valid <= 2:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model '.format(valid_loss_min, valid_loss))
        torch.save(model_test.state_dict(),'./model/Unet_D_' +
                                              str(epoch) + '_' + str(batch_size) + '/Unet_epoch_' + str(epoch)
                                              + '_batchsize_' + str(batch_size) + '.pth')
       # print(accuracy)
        if round(valid_loss, 4) == round(valid_loss_min, 4):
            print(i_valid)
            i_valid = i_valid+1
        valid_loss_min = valid_loss
        #if i_valid ==3:
         #   break

    #######################################################
    # Extracting the intermediate layers
    #######################################################

    #####################################
    # for kernals
    #####################################
    x1 = torch.nn.ModuleList(model_test.children())
    # x2 = torch.nn.ModuleList(x1[16].children())
     #x3 = torch.nn.ModuleList(x2[0].children())

    #To get filters in the layers
     #plot_kernels(x1.weight.detach().cpu(), 7)

    #####################################
    # for images
    #####################################
    x2 = len(x1)
    dr = LayerActivations(x1[x2-1]) #Getting the last Conv Layer

    img = Image.open(test_image)
    s_tb = data_transform(img)

    pred_tb = model_test(s_tb.unsqueeze(0).to(device)).cpu()
    pred_tb = F.sigmoid(pred_tb)
    pred_tb = pred_tb.detach().numpy()

    plot_kernels(dr.features, n_iter, 7, cmap="rainbow")

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    n_iter += 1

#######################################################
#closing the tensorboard writer
#######################################################

#writer1.close()

#######################################################
#if using dict
#######################################################

#model_test.filter_dict
