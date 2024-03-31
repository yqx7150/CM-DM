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
 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig6.png" width = "400" height = "450">  </div>

Visual Representation of different procedures of the proposed CM-DM. New k-space objects are constructed employing different high-frequency prior extractors which restrict the diffusion process to form “Weight-K-Space” and “Mask-K-Space” respectively, as shown in the blue and red parts of the first row. Yellow part of the second row illustrates that the input data is first constructed into the form corresponding to “Weight-K-Space” and “Mask-K-Space”. Subsequently, data is amalgamated and reconstructed either in series or parallel manner, followed by the introduction of a low-rank operator to further enhance the overall reconstruction effectiveness.

<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

Visualization of the underlying features of high-frequency operators. Yellow line represents underlying features in “Weight-K-Space” and the blue line exhibits features corresponding to different kernels of “Mas-K-Space”. Meanwhile, red line shows the correlation of different feature maps.

## Comparisons with State-of-the-arts.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>


## Additional Experiments with Latest Diffusion Models.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

## Additional Experiments with Latest Diffusion Models.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

## Diffusion Models in High-frequency Domain
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>


## Performance of Preserving Pathological Regions
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>


## D.	Convergence Analysis and Computational Cost
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>















