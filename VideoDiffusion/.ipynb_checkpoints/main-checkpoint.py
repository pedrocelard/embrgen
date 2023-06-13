import torch
import os
import sys
import numpy as np

from skvideo import io 
from torchsummary import summary 
from einops import rearrange

from video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer


model = Unet3D(
    dim = 64, 
    channels = 1,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    channels = 1,
    num_frames = 20,
    timesteps = 200, 
    loss_type = 'l1'
).cuda()

data = ""

try:
    trainer = Trainer(
        diffusion,
        data,
        train_batch_size = 1,
        num_frames = 20,
        train_lr = 1e-4,
        save_and_sample_every = 1000,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        ema_decay = 0.995,
        amp = True 
    )
    trainer.train()
except Exception as e:
    print('An exceptional thing happed - %s' % e)
    f = open('log.txt', 'w')
    f.write('An exceptional thing happed - %s' % e)
    f.close()