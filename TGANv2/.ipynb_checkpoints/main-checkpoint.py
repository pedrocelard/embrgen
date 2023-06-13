import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from tqdm.gui import tqdm
from scipy.ndimage.interpolation import shift
from pathlib import Path
from skvideo import io
from natsort import natsorted
from torchsummary import summary

from MovingEmbryo import MovingEmbryo

from models.tganv2_gen import Generator_CLSTM
from models.tganv2_dis import DisMultiResNet


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image, cmap='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def genSamples(g, n=4, e=1):
    img_size = 256
    
    with torch.no_grad():
        s = g(torch.rand((n**2, 256), device='cuda')*2-1,
              test=True).cpu().detach().numpy()
    out = np.zeros((1, 20, img_size*n, img_size*n))

    for j in range(n):
        for k in range(n):
            out[:, :, img_size*j:img_size*(j+1), img_size*k:img_size*(k+1)] = s[j*n + k, 0, :, :, :]

    out = out.transpose((1, 2, 3, 0))
    out = (np.concatenate([out, out, out], axis=3)+1) / 2 * 255
    io.vwrite(f'tganv2moving/gensamples_id{e}.gif', out)


def subsample_real(h, frames=4):
    h = h[:, np.random.randint(min(frames, h.shape[1]))::frames]
    return h


def full_subsample_real(h, frames=4):
    out = []
    for i in range(4):
        if i:
            out.append(subsample_real(out[i-1], frames=frames))
        else:
            out.append(h)

    for i in range(4):
        for j in range(3-i):
            out[i] = F.avg_pool3d(out[i], kernel_size=(1, 2, 2))
    return out


def zero_centered_gp(real_data, pr):
    gradients = torch.autograd.grad(outputs=pr, inputs=real_data,
                                    grad_outputs=torch.ones_like(pr),
                                    create_graph=True, retain_graph=True)

    return sum([torch.sum(torch.square(g)) for g in gradients])


def dataGen():
    while True:
        for d in loader:
            yield d
            # yield tensor_chromatic_aberration(torch.cat(d, dim=1), displacement)
            
def train(epochs, batch_size, lambda_val, colors, img_size, log_file):
    zt_dim_size = int(img_size/16)
    
    dg = dataGen()
    dis = DisMultiResNet(channels=[32, 32, 64, 128, 256], colors=colors).cuda()
    gen = Generator_CLSTM(
        tempc=256,
        zt_dim=4,
        upchannels=[128],
        subchannels=[64, 32, 32],
        n_frames=20,
        colors=colors
    ).cuda()
    
    disOpt = torch.optim.Adam(dis.parameters(), lr=5e-5, betas=(0, 0.9))
    genOpt = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

    try:
        for epoch in tqdm(range(epochs)):

            disOpt.zero_grad()
            
            if(colors==3):
                real = torch.cat(next(dg), dim=1).cuda().permute(0,1,4,2,3)
            else:
                real = torch.cat(next(dg), dim=1).cuda().unsqueeze(2)
                
            real = real.to(dtype=torch.float32) / 255 * 2 - 1
            real = full_subsample_real(real)
            
            for i in real:
                i.requires_grad = True
            pr = dis(real)
            dis_loss = zero_centered_gp(real, pr) * lambda_val
            
            with torch.no_grad():
                fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
                
            pf = dis(fake)
            dis_loss += torch.mean(F.softplus(-pr)) + torch.mean(F.softplus(pf))
            dis_loss.backward()
            disOpt.step()
            
            genOpt.zero_grad()
            
            fake = gen(torch.rand((batch_size, 256), device='cuda')*2-1)
            pf = dis(fake)
            
            gen_loss = torch.mean(F.softplus(-pf))
            gen_loss.backward()
            genOpt.step()
            
            # log results
            if epoch % 1000 == 0:
                print('Epoch', epoch, 'Dis', dis_loss.item(), 'Gen', gen_loss.item())
                l = open('log_'+log_file+'.txt', 'a')
                l.write('Epoch '+ str(epoch) + ' Dis '+ str(dis_loss.item())+ ' Gen '+ str(gen_loss.item())+"\n")
                l.close()

        torch.save({
                    'GEN': gen.state_dict(),
                }, './checkpoints/'+log_file+'.pth')
        
    except Exception as e:
        f = open('log.txt', 'w')
        f.write('An exceptional thing happed - %s' % e)
        f.close()
        
        
dataset = ""
size = 64
colors = 1
log_file = Path(dataset).name+"_"+str(size)
epochs = 100000
batch_size = 16
lambda_val = 0.5

data = MovingYeast(dataset, train=True, download=False, process=False, size=size)
loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True, drop_last=True)

train(epochs=epochs, 
      batch_size=batch_size, 
      lambda_val=lambda_val,
      colors=colors,
      img_size=size,
      log_file=log_file)