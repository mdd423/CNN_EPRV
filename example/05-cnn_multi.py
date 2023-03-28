#!/usr/bin/env python
# coding: utf-8
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
sys.path.append('..')

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
import src.models

import astropy.coordinates as coords
import astropy.time as at
from astropy.io import fits

import random
import numpy as np
import h5py as h5

torch.manual_seed(101101)
random.seed(101101)
np.random.seed(101101)

import argparse
def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--n_epochs',type=int,default=1)

    parser.add_argument('--s_in',type=int,default=43264)
    parser.add_argument('--s_h',type=int,default=64)

    parser.add_argument('--lr',type=np.double,default=0.001)
    parser.add_argument('--b1',type=np.double,default=0.9)
    parser.add_argument('--b2',type=np.double,default=0.999)

    return parser

if __name__ == '__main__':
    opt  = get_parser().parse_args()
    img_stack,rvs_stack,bcs_stack,tim_stack,ind_stack,address = src.models.load_sets('/scratch/mdd423/CNN_EPRV/data/peg51_256/raw/peg51_256/*02-28*')
    dataset = src.models.ND_Dataset(img_stack,bcs_stack,ind_stack)
    testdata,validdata = torch.utils.data.random_split(dataset,[23880,1000])

    n_cpu = 2
    dataloader = DataLoader(
        testdata,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    validloader = DataLoader(
        validdata,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # lr = 0.001
    # b1 = 0.9
    # b2 = 0.999

    s_c  = int(np.max(ind_stack))
    # s_in = 43264
    # s_h  = 64
    model     = src.models.RV_Model(s_c + 1,opt.s_in,opt.s_h,device).to(device)
    mse_loss  = torch.nn.MSELoss().double()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    thing = 0
    for parameter in model.chunk_models.parameters():
        thing += np.product(parameter.shape)

    thing2 = 0
    for parameter in model.feature_model.parameters():
        thing2 += np.product(parameter.shape)
    print('dense',thing/s_c)
    print('conv   ' ,thing2)
    print('dcr', thing/s_c/thing2)


    valiter = itertools.cycle(validloader)
    start_time = time.time()
    batch = next(valiter)
    y = model(batch['img'].to(device),batch['indices']).squeeze()
    print('batch time: ', time.time() - start_time)
    print(torch.cuda.memory_summary())


    subprocess.run(['nvidia-smi'])

    n_epochs = opt.n_epochs
    train_loss = []
    valid_loss = []

    valiter = itertools.cycle(validloader)
    b_avg = 0.0
    e_avg = 0.0

    start_t = time.time()

    # directory, tail = path.split(filename)
    for j,epoch in enumerate(range(opt.n_epochs)):

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
                            "[Epoch %d/%d | Batch %d/%d | TL: %f | VL: %f | BT: %s | ET: %s | RT: %s]"
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
            modelpath = '/scratch/mdd423/CNN_EPRV/models/rv_model_multi_{}_{}_bcs.model'.format(j,n_epochs)
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
