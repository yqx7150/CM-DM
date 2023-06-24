import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
from skimage.measure import compare_psnr,compare_ssim
import time
import math
import torch
#from scipy import linalg

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
    size_mtx = mtx.shape #(63001, 36, 8)
    sx = size_data[0] # 256
    sy = size_data[1] # 256
    sz = size_mtx[2] # 8
    
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

def sake(K_input, Ksample, mask, ksize, wnthresh, sakeIter):

    size_data = (K_input).shape
    Krec = np.copy(K_input)#np.zeros((Ksample.shape),dtype=np.complex64)#np.copy(Ksample)
     
    for n in range(sakeIter):
        print('==sake== ',n)
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
        
        #Kout = np.copy(Krec)
        Image_Rec = ifft2c(Krec)
        #print(np.max(abs(Image_Rec)))
        '''
        Image_sos = np.sqrt(np.sum(np.square(np.abs(Image_Rec)),axis=2)) 
        Image_sos = Image_sos/np.max(np.abs(Image_sos))
        psnr=compare_psnr(255*abs(Image_sos),255*abs(ori_sos),data_range=255)
        ssim=compare_ssim(abs(Image_sos),abs(ori_sos),data_range=1)
        print('psnr: ',psnr,'ssim: ',ssim) 
        '''
    return Image_Rec, Krec

