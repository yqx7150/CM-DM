import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import time
import math
import torch
import cv2
#from scipy import linalg

def compare_hfen(rec,ori):
    operation = np.array(io.loadmat("./mask/loglvbo.mat")['h1'],dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord = 'fro')
    return hfen

def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)

def fft2c(x):
    size = (x).shape
    fctr = size[0]*size[1]
    Kdata = np.zeros((size),dtype=np.complex64)
    for i in range(size[2]):
        Kdata[:,:,i] = (1/np.sqrt(fctr))*np.fft.fftshift(np.fft.fft2(x[:,:,i]))
    return Kdata
        
def ifft2c(kspace):
    size = (kspace).shape
    fctr = size[0]*size[1]
    Image = np.zeros((size),dtype=np.complex64)
    for i in range(size[2]):
        Image[:,:,i] = np.sqrt(fctr)*np.fft.ifft2(kspace[:,:,i])
    return Image    

def im2row(im,winSize):
    size = (im).shape
    out = np.zeros(((size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),winSize[0]*winSize[1],size[2]),dtype=np.complex64)
    count = -1
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1                 
            temp1 = im[x:(size[0]-winSize[0]+x+1),y:(size[1]-winSize[1]+y+1),:]
            temp2 = np.reshape(temp1,[(size[0]-winSize[0]+1)*(size[1]-winSize[1]+1),1,size[2]],order = 'F')
            out[:,count,:] = np.squeeze(temp2) # MATLAB reshape          
            
    return out

def row2im(mtx,size_data,winSize):
    size_mtx = mtx.shape
    sx = size_data[0]
    sy = size_data[1]
    sz = size_mtx[2]
    
    res = np.zeros((sx,sy,sz),dtype=np.complex64)
    W = np.zeros((sx,sy,sz),dtype=np.complex64)
    out = np.zeros((sx,sy,sz),dtype=np.complex64)
    count = -1
    
    for y in range(winSize[1]):
        for x in range(winSize[0]):
            count = count + 1
            res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = res[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + np.reshape(np.squeeze(mtx[:,count,:]),[sx-winSize[0]+1,sy-winSize[1]+1,sz],order = 'F')  
            W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] = W[x : sx-winSize[0]+x+1 ,y : sy-winSize[1]+y+1 ,:] + 1
            

    out = np.multiply(res,1./W)
    return out

def sake(K_input, Ksample, mask, ksize, wnthresh, sakeIter, ori):
    PSNR_all = []
    SSIM_all = []
    HFEN_all = []
    #size_data = (K_input).shape
    Krec = np.copy(K_input)#np.zeros((Ksample.shape),dtype=np.complex64)#np.copy(Ksample)   
    Krec = Krec.transpose(1, 2, 0)
    size_data = (Krec).shape
    Ksample = Ksample.transpose(1, 2, 0)
    mask = mask.transpose(1, 2, 0)
    ori = ori.transpose(1, 2, 0)
    
    for n in range(sakeIter):
        #print('==sake== ',n)
        temp = im2row(Krec,ksize)          
        size_temp = temp.shape
        A = np.reshape(temp,[size_temp[0],size_temp[1]*size_temp[2]],order = 'F') # max: 14.925017 (matlab:14.9250) (63001, 288)
  
        #=============================================== SVD
        A  = torch.tensor(A,dtype=torch.complex64)
        U,S,V = torch.svd(A)
        S = torch.diag(S)
       
        U = np.array(U,dtype=np.complex64)
        S = np.array(S,dtype=np.complex64)
        V = np.array(V,dtype=np.complex64)           
        #===============================================
        
        uu = U[:,0:math.floor(wnthresh*ksize[0]*ksize[1])] #(63001, 64)
        ss = S[0:math.floor(wnthresh*ksize[0]*ksize[1]),0:math.floor(wnthresh*ksize[0]*ksize[1])] #(64, 64)
        vv = V[:,0:math.floor(wnthresh*ksize[0]*ksize[1])]  #(64, 288)  

        A = np.dot(np.dot(uu,ss),vv.T) 
        A = np.reshape(A,[size_temp[0],size_temp[1],size_temp[2]],order = 'F')
        
        kcomplex = row2im(A,size_data,ksize)
    
        #print(np.max(abs(kcomplex)),' ',np.min(abs(kcomplex)))
        
        for ii in range(size_temp[2]):
            Krec[:,:,ii] = kcomplex[:,:,ii]*(1-mask[:,:,ii]) + Ksample[:,:,ii]

        Image_Rec = ifft2c(Krec)
        
        for i in range(16):          
          eval_ori = (abs(ori[:, :, i]))/np.max(abs(ori[:, :, i]))
          eval_rec = (abs(Image_Rec[:, :, i]))/np.max(abs(Image_Rec[:, :, i]))
          psnr_ori_rec = PSNR(255*eval_ori, 255*eval_rec, data_range=255)
          ssim_ori_rec = SSIM(eval_ori, eval_rec, data_range=1)
          hfen_ori_rec = compare_hfen(eval_rec, eval_ori)
          PSNR_all.append(psnr_ori_rec)
          SSIM_all.append(ssim_ori_rec)
          HFEN_all.append(hfen_ori_rec)
          write_images(eval_ori, osp.join('./result/'+'ori_'+str(i)+'.png'))
          write_images(eval_rec, osp.join('./result/'+'rec_'+str(i)+'.png'))
        print('Average psnr: ', sum(PSNR_all)/len(PSNR_all))
        print('Average ssim: ', sum(SSIM_all)/len(SSIM_all))
        print('Average hfen: ', sum(HFEN_all)/len(HFEN_all)) 

    return Image_Rec, Krec

