#@title Autoload all modules
#%load_ext autoreload
#%autoreload 2

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import scipy.io as io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
import os.path as osp
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
from skimage.measure import compare_psnr,compare_ssim
import cv2

def write_kdata(Kdata,name):
    temp = np.log(1+abs(Kdata))    
    plt.axis('off')
    plt.imshow(abs(temp),cmap='gray')
    plt.savefig(osp.join('./result/',name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')

def write_Data(model_num,step,psnr,ssim):
    filedir="result.txt"
    with open(osp.join('./result/',filedir),"a+") as f:
        f.writelines(str(model_num)+' '+str(step)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')

def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
  
def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y
    
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)    

# @title Load the score-based model

sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE']

if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  #ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  model_num = 'checkpoint_100.pth'
  ckpt_filename = './exp/checkpoints/old/checkpoint_5.pth'  # 修改checkpoint
  
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) ###################################  sde
  #sde = VESDE(sigma_min=0.01, sigma_max=10, N=100) ###################################  sde
  sampling_eps = 1e-5


batch_size =   1 
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0 

#sigmas = mutils.get_sigmas(config) # 根据最大最小值创建不同噪声等级  （后面没用到）
scaler = datasets.get_data_scaler(config) # centered = False 不改变数据值域
inverse_scaler = datasets.get_data_inverse_scaler(config) # centered = False 不改变数据值域
score_model = mutils.create_model(config) # 建立模型

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())


for pp in range(1):
    write_psnr=0
    write_ssim=0
    
    pic= pp + 1
    print('picture:',pic)
    if pic<10:
        file_path='/home/lqg/桌面/TZJ_SDE/input_data/SIAT_test_image31/test_data_0'+str(pic)+'.mat'
    else:
        file_path='/home/lqg/桌面/TZJ_SDE/input_data/SIAT_test_image31/test_data_'+str(pic)+'.mat'
        
    ori_data = np.zeros([256,256],dtype=np.complex64)
    ori_data = io.loadmat(file_path)['Img']
    ori_data = ori_data/np.max(abs(ori_data))

    mask = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/mask/random_mask_r3_256.mat')['mask']  
    #mask = io.loadmat('./input_data/mask/poisson/2.mat')['mask'] 
    #mask = io.loadmat('./input_data/mask/random2D/2.mat')['mask']   
    mask_DCused = mask
    
    weight = io.loadmat('/home/lqg/桌面/TZJ_SDE/input_data/weight1.mat')['weight']    
    
    Kdata=np.fft.fftshift(np.fft.fft2(ori_data))
    ori_data = np.fft.ifft2(Kdata)
    
    Ksample=np.multiply(mask,Kdata)
    Zeorfilled_data=np.fft.ifft2(Ksample)   
    k_w = k2wgt(Ksample,weight)

    psnr_zero=compare_psnr(255*abs(Zeorfilled_data),255*abs(ori_data),data_range=255)
    ssim_zero=compare_ssim(abs(Zeorfilled_data),abs(ori_data),data_range=1)
    print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)

    # PC sampling 网络参数设置
    img_size = config.data.image_size
    channels = config.data.num_channels
    shape = (batch_size, channels, img_size, img_size)
    predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"]
    corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"]
    snr = 0.075#0.075  #0.16
    n_steps = 1
    probability_flow = False 
    sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                          inverse_scaler, snr, n_steps=n_steps,
                                          probability_flow=probability_flow,
                                          continuous=config.training.continuous,
                                          eps=sampling_eps, device=config.device)    
  
    

    #x_input = sde.prior_sampling(shape).to(config.device) #初始化输入
    #x_input = x_input.cpu().numpy()
    x_input = np.random.uniform(-1,1,size=(1,3,256,256))
    x_input[0,0,:,:] = np.real(k_w)
    x_input[0,1,:,:] = np.imag(k_w)
    x_input[0,2,:,:] = np.real(k_w)
    x_input = torch.from_numpy(x_input).to(config.device)
    x_input = x_input.type(torch.cuda.FloatTensor)
    max_psnr = 0
    max_ssim = 0
            
    for iii in range(1):      
      x, n = sampling_fn(score_model,x_input) # 网络输出 
      x = x.detach().cpu().numpy() # (1,3,256,256)          

      kw_real = (x[0,0,:,:]+x[0,2,:,:])/2
      kw_imag = x[0,1,:,:]
      k_w = kw_real+1j*kw_imag
          
      k_complex = wgt2k(k_w,weight,Ksample)
      k_complex2 = Ksample + k_complex*(1-mask_DCused)

      print('k_w ',np.max(abs(k_w)),' ',np.min(abs(k_w)))
      #print(np.max(abs(k_complex)),' ',np.min(abs(k_complex)))
      #temp = np.log(1+abs(k_complex))
      #write_images(abs(temp),osp.join('./result/'+'k_complex'+str(iii)+'.png')
      
      rec_Image = np.fft.ifft2(k_complex2)
      '''
      plt.ion()
      plt.imshow(abs(rec_Image),cmap='gray')
      plt.axis('off')
      plt.pause(0.5)
      '''
      k_w = k2wgt(k_complex2,weight)
      x_input = x_input.detach().cpu().numpy() 
      x_input[0,0,:,:] = np.real(k_w)
      x_input[0,1,:,:] = np.imag(k_w)
      x_input[0,2,:,:] = np.real(k_w)
      x_input = torch.from_numpy(x_input).to(config.device)        
      x_input = x_input.type(torch.cuda.FloatTensor)

      write_images(abs(rec_Image),osp.join('./result/'+'Rec_image'+str(iii)+'.png'))
      #io.savemat(osp.join('./result/'+'Rec_image'+str(iii)+'.mat'),{'rec_Image':rec_Image})
   
      psnr = compare_psnr(255*abs(rec_Image),255*abs(ori_data),data_range=255)
      ssim = compare_ssim(abs(rec_Image),abs(ori_data),data_range=1)
      print("step:{}".format(iii),' PSNR:', psnr,' SSIM:', ssim)
      
      if max_ssim<=ssim:
        max_ssim = ssim
      if max_psnr<=psnr:
        max_psnr = psnr
        write_Data(model_num,iii,max_psnr,max_ssim)

