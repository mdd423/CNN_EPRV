import torch
import torch.nn as nn
import torch.nn.functional as F

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

#             temp = int()
#             print(y1.index_select(0,torch.where(indices==index)[0].to(self.device)))
#             print(torch.where(indices==index)[0])
#             print(y2.shape)
            y2[torch.where(indices==index)[0]] = self.chunk_models[index.item()](y1.index_select(0,torch.where(indices==index)[0].to(self.device))).squeeze()

        return y2

import pickle
def save(filename,model):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

def load(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        model = pickle.load(input)
        return model


def load_sets(dirqueue):
    all_directories = glob.glob(dirqueue)
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

    address = []
    for i,filename in enumerate(files):
        for j,hdu in enumerate(['hdu_1','hdu_2']):
    #     filename             = files1[0]

    #         ds                   = h5.File(filename,'r')
    #         img_stack, rvs_stack, bcs_stack, tim_stack = h5_to_array(ds,target,location,hdu_num=hdu)
            dir_name, tailname = path.split(filename)
            address.append(tailname + '_' + hdu)
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


    return img_stack,rvs_stack,bcs_stack,tim_stack,address
