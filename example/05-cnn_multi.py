#!/usr/bin/env python
# coding: utf-8

# <h1>05-CNN for Multiple chunks of data</h1>

# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import sys
import time
import datetime
import itertools
import os
import os.path as path
import subprocess
import pickle
import glob

import astropy.coordinates as coords
import astropy.time as at
from astropy.io import fits

import random
import numpy as np
import h5py as h5

torch.manual_seed(101101)
random.seed(101101)
np.random.seed(101101)

def save(filename,model):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model

def h5_to_array(ds,target,location,hdu_num='hdu_1'):
    rvs_stack = []
    bcs_stack = []
    tim_stack = []

    img_stack = np.empty((0,3,256,256))
    for visit_name,vist_info in ds['visits'].items():
#         for hdu_num in ds['images'][visit_name].keys():
        img   = np.array(ds['images'][visit_name][hdu_num])
#
        if np.sum(img.shape) == (256*2):

            rvs_stack += [np.double(vist_info.attrs['ESO DRS CCF RVC'])]
            temp_time = at.Time(visit_name.split('HARPS.')[1])
            tim_stack += [temp_time]
            bcs_stack += [target.radial_velocity_correction(obstime=temp_time, location=location).to('km/s').value]
            all_flats = np.empty([0,256,256])
            for key in vist_info.attrs.keys():
                if vist_info.attrs[key] == 'FLAT':
                    temp = np.array(ds['images'][key][hdu_num])[None,...]
                    all_flats = np.append(all_flats, temp,axis=0)
                if vist_info.attrs[key] == 'THAR_THAR':
                    cali = np.array(ds['images'][key][hdu_num])
#                 temp = crop_again(np.stack((img,np.median(all_flats,axis=0),cali)))
            temp = np.stack((img,np.median(all_flats,axis=0),cali))[None,...]
#                 print(temp.shape)
            img_stack = np.append(img_stack,temp,axis=0)
        else:
            print(visit_name)
    return img_stack, np.array(rvs_stack), np.array(bcs_stack), np.array(tim_stack)

class ND_Dataset(Dataset):
    def __init__(self, imgs,rvs,indices):
        self.img_stack = imgs
        self.rvs_stack = rvs
        self.indices   = indices
        self.type      = torch.Tensor

    def __getitem__(self, index):

        return {'img': self.type(self.img_stack[index,...]).double(),
                'rvs': np.double(self.rvs_stack[index]),
                'indices': self.indices[index]}


    def __len__(self):

        return len(self.rvs_stack)

class DownSizeNet(nn.Module):
    def __init__(self, in_size, out_size, kernel_size = 3, stride=1, padding=1, leaky_slope=0.2):
        super(DownSizeNet, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, stride=stride, padding=padding, bias=False).double()]
        layers.append(nn.MaxPool2d(3, stride=stride).double())
        layers.append(nn.LeakyReLU(leaky_slope).double())

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)

class ChunkNet(nn.Module):
    def __init__(self,s_in,s_h):
        super(ChunkNet, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(s_in,s_h), nn.ReLU()).double()

        self.final = nn.Linear(s_h,1).double()

    def forward(self,x):
        d1 = self.dense1(x)

        return self.final(d1)

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.down1 = DownSizeNet(3,   64,  stride=1)
        self.down2 = DownSizeNet(64,  64,  stride=1, padding=0)
        self.down3 = DownSizeNet(64,  128, stride=1, padding=0, kernel_size=5)
        self.down4 = DownSizeNet(128, 256, stride=2, padding=0, kernel_size=5)
        self.down5 = DownSizeNet(256, 256, stride=2, padding=0, kernel_size=5)


    def forward(self, x):
        # Propogate noise through fc layer and reshape to img shape
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
#         d6 = self.down6(d5)

        return d5

class RV_Model(nn.Module):
    def __init__(self,s_c,s_in,s_h,device):
        super(RV_Model, self).__init__()
        self.device = device

        self.chunk_models = nn.ModuleList([ChunkNet(s_in,s_h).double().to(device) for i in range(s_c)]).to(device)
        self.feature_model = FeatureNet().double().to(device)

    def forward(self,x,indices):
        y1 = self.feature_model(x)
        y1 = torch.flatten(y1,1)
        y2 = torch.empty((y1.shape[0])).to(self.device).double()
        # instead of looping through all indices passed
        # only loop through unique ones so that chunk models
        # can be batch-ran
        for i,index in enumerate(np.unique(indices).astype(int)):

            y2[torch.where(indices==index)[0]] = self.chunk_models[index.item()](y1.index_select(0,torch.where(indices==index)[0].to(self.device))).squeeze()

        return y2

if __name__ == '__main__':

    files1 = glob.glob('/scratch/mdd423/CNN_EPRV/data/peg51_256/*.h5')

    all_directories = glob.glob('/scratch/mdd423/CNN_EPRV/data/peg51_256/raw/peg51_256/*02-28*')
    files = []
    for indiv in all_directories:
        files += glob.glob(indiv + '/*.h5')


    location = coords.EarthLocation.of_site('La Silla Observatory')
    target   = coords.SkyCoord.from_name('51PEG')

    iterations = 0
    img_stack = np.empty((0,3,256,256))
    rvs_stack = np.empty((0))
    bcs_stack = np.empty((0))
    tim_stack = np.empty((0))
    ind_stack = np.empty((0))

    start_time = time.time()
    total = 2*len(files)
    for i,filename in enumerate(files):
        for j,hdu in enumerate(['hdu_1','hdu_2']):
    #     filename             = files1[0]

    #         ds                   = h5.File(filename,'r')
    #         img_stack, rvs_stack, bcs_stack, tim_stack = h5_to_array(ds,target,location,hdu_num=hdu)
            dir_name, tailname = path.split(filename)
            tailname = tailname[:-3]
            imgname = path.join(dir_name, tailname + hdu + '_img.nda')
            rvsname = path.join(dir_name, tailname + hdu + '_rvs.nda')
            bcsname = path.join(dir_name, tailname + hdu + '_bcs.nda')
            timname = path.join(dir_name, tailname + hdu + '_tim.nda')

            img_stack = np.append(img_stack, load(imgname),axis=0)
            rvs_stack = np.append(rvs_stack, load(rvsname),axis=0)
            bcs_stack = np.append(bcs_stack, load(bcsname),axis=0)
            tim_stack = np.append(tim_stack, load(timname),axis=0)

            ind_stack = np.append(ind_stack, iterations*np.ones(bcs_stack.shape,dtype=int))
            iterations += 1

            if (i % 5) == 0:
                load_time = time.time() - start_time
                n_batches = (i*2 + j + 1)
                load_avg  = load_time/n_batches
                remaining = (total - n_batches) * load_avg
                sys.stdout.write('\r[ Loaded: {}/{} | Avg Time: {} | Remaining: {} ]'.format(\
                                                                                           n_batches,total,
                                                                                           load_avg,
                                                                                           str(datetime.timedelta(seconds=remaining))
                                                                                          ))
    dataset = ND_Dataset(img_stack,bcs_stack,ind_stack)

    testdata,validdata = torch.utils.data.random_split(dataset,[23880,1000])

    batch_size = 64
    n_cpu = 2
    dataloader = DataLoader(
        testdata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    validloader = DataLoader(
        validdata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    lr = 0.001
    b1 = 0.9
    b2 = 0.999

    s_c  = int(np.max(ind_stack))
    s_in = 43264
    s_h  = 64
    s_out= 64
    model     = RV_Model(s_c,s_in,s_h,device).to(device)
    mse_loss  = torch.nn.MSELoss().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

    thing = 0
    for parameter in model.chunk_models.parameters():
        thing += np.product(parameter.shape)

    thing2 = 0
    for parameter in model.feature_model.parameters():
        thing2 += np.product(parameter.shape)
    print('dense',thing/165)
    print('conv   ' ,thing2)
    print('dcr', thing/165/thing2)


    valiter = itertools.cycle(validloader)
    start_time = time.time()
    batch = next(valiter)
    y = model(batch['img'].to(device),batch['indices']).squeeze()
    print('batch time: ', time.time() - start_time)


    print(torch.cuda.memory_summary())


    subprocess.run(['nvidia-smi'])

    n_epochs = 70
    train_loss = []
    valid_loss = []

    valiter = itertools.cycle(validloader)
    b_avg = 0.0
    e_avg = 0.0

    start_t = time.time()

    directory, tail = path.split(filename)
    for j,epoch in enumerate(range(n_epochs)):

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            y    = model(batch['img'].to(device),batch['indices']).squeeze()

            loss = mse_loss(y.double(),batch['rvs'].to(device).double())
            loss.backward()
            optimizer.step()
            b_time = time.time() - start_t
            b_avg  = b_time / (i + 1 + (j * len(dataloader)))

            if (i % 10) == 0:
                # Validation checkpoint every 5 batches
                valbatch = next(valiter)
                model.eval()
                with torch.no_grad():
                    y     = model(     valbatch['img'].to(device),valbatch['indices']).squeeze()

                    vloss = mse_loss(y,valbatch['rvs'].to(device).double())

                    r_time = b_avg * ((len(dataloader) * n_epochs) - (i + 1 + (j * len(dataloader))))
                    sys.stdout.write(
                            "\r[Epoch %d/%d | Batch %d/%d | TL: %f | VL: %f | BT: %s | ET: %s | RT: %s]"
                            % (
                                epoch,
                                n_epochs,
                                i,
                                len(dataloader),
                                loss.item(),
                                vloss.item(),
                                str(datetime.timedelta(seconds=b_avg)),
                                str(datetime.timedelta(seconds=e_avg)),
                                str(datetime.timedelta(seconds=r_time))
                            )
                    )

                model.train()

        train_loss.append(loss.item())
        valid_loss.append(vloss.item())
        if (j % 1) == 0:
            # Saving checkpoint every 10 epoches
            modelpath = '/scratch/mdd423/CNN_EPRV/models/rv_model_multi_{}_{}_{}_bcs.model'.format(tail[:-3],j,n_epochs)
            torch.save(model.state_dict(), modelpath)
        e_time = time.time() - start_t
        e_avg  = e_time/(j + 1)
    train_loss = np.array(train_loss)
    valid_loss = np.array(valid_loss)


    modelpath = '/scratch/mdd423/CNN_EPRV/models/rv_model_multi_{}_{}_bcs.model'.format(j,n_epochs)
    torch.save(model.state_dict(), modelpath)

    tlname = path.join(dir_name, tailname + '_multi_tl_bcs.nda')
    vlname = path.join(dir_name, tailname + '_multi_vl_bcs.nda')
    np.save(tlname,train_loss)
    np.save(vlname,valid_loss)
