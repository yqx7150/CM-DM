#@title Autoload all modules


from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
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
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import Score_SDE_demo2
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from Score_SDE_demo2 import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import os.path as osp

sde = 'VESDE' 
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs 
  model_num = 'checkpoint.pth'
  ckpt_filename_weight ='../exp_total/exp_weight/checkpoints/checkpoint_100.pth' 
  ckpt_filename_mask ='../exp_total/exp_mask/checkpoints/checkpoint_100.pth' 
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) 
  sampling_eps = 1e-5
batch_size = 20 
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0 
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model_weight = mutils.create_model(config)

optimizer = get_optimizer(config, score_model_weight.parameters())
ema = ExponentialMovingAverage(score_model_weight.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model_weight, ema=ema)
state = restore_checkpoint(ckpt_filename_weight, state, config.device)
ema.copy_to(score_model_weight.parameters())
score_model_mask = mutils.create_model(config)
optimizer = get_optimizer(config, score_model_mask.parameters())
ema = ExponentialMovingAverage(score_model_mask.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model_mask, ema=ema)
state = restore_checkpoint(ckpt_filename_mask, state, config.device)
ema.copy_to(score_model_mask.parameters())
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor 
corrector = LangevinCorrector 
snr = 0.075
n_steps =  1
probability_flow = False 
sampling_fn = Score_SDE_demo2.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model_weight, score_model_mask)


