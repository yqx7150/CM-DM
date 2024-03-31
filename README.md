# CM-DM

**Paper**: CM-DM: Correlated and Multi-frequency Diffusion Modeling for Highly Under-sampled MRI Reconstruction

**Authors**: Yu Guan, Chuanming Yu, Zhuoxu Cui, Huilin Zhou*, Qiegen Liu*   

IEEE Transactions on Medical Imaging, https://ieeexplore.ieee.org/document/10478958.  

Date : 25-March-2024  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2024, School of Mathematics and Computer Sciences, Nanchang University.  


Given the obstacle in accentuating the reconstruction accuracy for diagnostically significant tissues, most existing MRI reconstruction methods perform targeted reconstruction of the entire MR image without considering fine details, especially when dealing with highly under-sampled images. Therefore, a considerable volume of efforts has been directed towards surmounting this challenge, as evidenced by the emergence of numerous methods dedicated to preserving high-frequency content as well as fine textural details in the reconstructed image. In this case, exploring the merits associated with each method of mining high-frequency information and formulating a reasonable principle to maximize the joint utilization of these approaches will be a more effective solution to achieve accurate reconstruction. Specifically, this work constructs an innovative principle named Correlated and Multi-frequency Diffusion Model (CM-DM) for highly under-sampled MRI reconstruction. In essence, the rationale underlying the establishment of such principle lies not in assembling arbitrary models, but in pursuing the effective combinations and replacement of components. It also means that the novel principle focuses on forming a correlated and multi-frequency prior through different high-frequency operators in the diffusion process. Moreover, multi-frequency prior further constraints the noise term closer to the target distribution in the frequency domain, thereby making the diffusion process converge faster. Experimental results verify that the proposed method achieved superior recon-struction accuracy, with a notable enhancement of ap-proximately 2dB in PSNR compared to state-of-the-art methods.

## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/fig1.png" width = "1000" height = "450">  </div>

Visual Representation of different procedures of the proposed CM-DM. New k-space objects are constructed employing different high-frequency prior extractors which restrict the diffusion process to form “Weight-K-Space” and “Mask-K-Space” respectively, as shown in the blue and red parts of the first row. Yellow part of the second row illustrates that the input data is first constructed into the form corresponding to “Weight-K-Space” and “Mask-K-Space”. Subsequently, data is amalgamated and reconstructed either in series or parallel manner, followed by the introduction of a low-rank operator to further enhance the overall reconstruction effectiveness.

<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/fig. 2.png" width = "600" height = "450"> </div>

Visualization of the underlying features of high-frequency operators. Yellow line represents underlying features in “Weight-K-Space” and the blue line exhibits features corresponding to different kernels of “Mas-K-Space”. Meanwhile, red line shows the correlation of different feature maps.

## Comparisons with State-of-the-arts.
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/table 3.png"> </div>
PSNR, SSIM, and MSE (*E-4) comparison with state-of-the-art methods under poisson, 2D random, and uniform sampling patterns with different acceleration factors.
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f3.png"> </div>
Reconstruction of the T1-weighted Brain at random sampling of R=12. From left to right: Full-sampled, Under-sampled, reconstruction by SAKE, P-LORAKS, EBMRec, HGGDP, and CM-DM. The second row shows the enlarged view of the ROI region (indicated by the yellow box in the first row), and the third row shows the error map of the reconstruction. Yellow numbers in the upper right corner indicate PSNR (dB), SSIM and MSE (*E-4), respectively.

<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f4.png"> </div>
Complex-valued reconstruction results at R=10 using uniform sampling with 8 coils. From left to right: Full-sampled, Under-sampled, reconstruction by SAKE, P-LORAKS, EBMRec, HGGDP, and CM-DM. The second row shows the enlarged view of the ROI region (indicated by the yellow box in the first row), and the third row shows the error map of the reconstruction. The values in the corner are PSNR (dB), SSIM and MSE (*E-4) values of each slice. 


## Additional Experiments with Latest Diffusion Models.
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f5.png"> </div>
Reconstruction results under uniform under-sampled at R=12. The values in the corner are each slice’s PSNR/SSIM/MSE values. Second and third rows illus-trate the enlarged and error views, respectively. From left to right: Full-sampled, under-sampled, reconstruction by CSGM-MRI, and CM-DM.



## Diffusion Models in High-frequency Domain
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f6.png"> </div>
Complex-valued reconstruction results at R=15 using Cartesian sampling with 12 coils. From left to right: Full-sampled, under-sampled, reconstruction by HFS-SDE, WKGM, and CM-DM. The second row shows the enlarged view of the ROI region (indicated by the yellow box in the first row), and the third row shows the error map of the reconstruction.




## Performance of Preserving Pathological Regions
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f7.png"> </div>

Reconstruction results using E2E-Varnet and CM-DM at R=10 of the eq-uispaced mask. Second and third rows illustrate the magnified views corre-sponding to pathological regions and error views, respectively. The color bar of the error images is at the right of the figure.




## Convergence Analysis and Computational Cost
<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/f8.png"> </div>
Convergence curves of WKGM and CM-DM in terms of PSNR and SSIM versus the iteration number when reconstructing the brain image from 1/8 sampled data under random sampling pattern.

<div align="center"><img src="https://github.com/yqx7150/CM-DM/blob/main/samples/t5.png"> </div>


## Other Related Projects
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S0925231221000990) [<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)

  * Complex-valued MRI data from SIAT--test31 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31)
  * More explanations with regard to the MoDL test datasets, we use some data from the test dataset in "dataset.hdf5" file, where the image slice numbers are 40,48,56,64,72,80,88,96,104,112(https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view)
  * DDP Method Link [<font size=5>**[DDP Code]**</font>](https://github.com/kctezcan/ddp_recon)
  * MoDL Method Link [<font size=5>**[MoDL code]**</font>](https://github.com/hkaggarwal/modl)
  * Complex-valued MRI data from SIAT--SIAT_MRIdata200 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT_MRIdata200)  
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-singlecoil [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-singlecoil)   
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-12coils [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-12coils)    
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
    
 * One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/10158730)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HKGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)
    
* Lens-less imaging via score-based generative model (基于分数匹配生成模型的无透镜成像方法)
[<font size=5>**[Paper]**</font>](https://www.opticsjournal.net/M/Articles/OJf1842c2819a4fa2e/Abstract)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/LSGM)









