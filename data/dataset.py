import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

class NUSNMRDataset(Dataset):
    def __init__(self, num_samples, split='train', need_LR=False):
        super(NUSNMRDataset,self).__init__()
        self.num_samples = num_samples
        self.split = split
        self.mul = 10000
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        NUS_FID, NOISE_input, GT_label = self.__gen_signal__(idx)
        power_of_10 = int(np.log10(np.max([np.real(NOISE_input), np.imag(NOISE_input)]))) + 1 if np.max([np.real(NOISE_input), np.imag(NOISE_input)]) != 0 else 1
        NUS_FID = torch.complex(torch.from_numpy(NUS_FID.real).float(), torch.from_numpy(NUS_FID.imag).float())
        NOISE_input = torch.complex(torch.from_numpy(NOISE_input.real).float(), torch.from_numpy(NOISE_input.imag).float()) / 10**power_of_10
        GT_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / 10**power_of_10

        # LR = torch.cat((NUS_FID.real.unsqueeze(0), NUS_FID.imag.unsqueeze(0)), dim=0)
        LR = NUS_FID.unsqueeze(0)
        SR = torch.cat((NOISE_input.real.unsqueeze(0), NOISE_input.imag.unsqueeze(0)), dim=0)
        HR = torch.cat((GT_label.real.unsqueeze(0), GT_label.imag.unsqueeze(0)), dim=0)

        return {'LR': LR, 'HR': HR, 'SR': SR, 'Index': idx}
    
    def __gen_signal__(self, idx):
        # np.random.seed(idx)
        # Define simulation parameters
        dim = 2
        
        # if self.flag % 60 < 20:
        #     N = 64
        # elif self.flag % 60 < 40 and self.flag % 60 >= 20:
        #     N = 128
        # else:
        #     N = 256
        # self.flag += 1
        N = 256

        # sample_rate = np.random.uniform(self.sample_rate, 0.3)
        t = np.arange(N) # time axis
        max_J = 10

        # Generate FID signals
        J = np.random.randint(1, max_J+1, size=(dim,1))   # Random number of harmonics
        mask = np.zeros((dim, max_J))
        mask[np.arange(dim), J.ravel()-1] = 1  # TODO
        mask = np.cumsum(mask, axis=1)

        ph = np.random.uniform(0.0, 2*np.pi, size=(dim, max_J))  # Random phase  # TODO
        A = np.random.uniform(0.05, 1.0, size=(dim, max_J))  # Random amplitude
        w = np.random.uniform(0.01, 0.99, size=(dim, max_J))  # Random frequency
        sgm = np.random.uniform(10, 179.2, size=(dim, max_J))  # Random relaxation time
        t = np.arange(N) # Time axis

        A = np.multiply(A, mask)
        x = A[..., None] * np.exp(1j * ph[..., None]) * np.exp(-t / sgm[..., None]) * np.exp(1j*2*np.pi*w[..., None]*t)
        xn_unit = np.matmul(x[0][:, :, np.newaxis], x[1][:, np.newaxis])
        clean_xn = np.sum(xn_unit, axis=0)

        # Add noise to FID signals
        noise_scale = 1e-4
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=(N, N))
        xx = noise + clean_xn

        GT_label = np.fft.fft2(xx)

        # Generate undersampling mask 
        if self.split == 'train':    
            Mask = np.load('./data/poisson_gap_mask/'+str(N)+'/Mask_poisson_'+str(idx%5)+'.npy')  # TODO
        elif self.split == 'valid':
            Mask = np.load('./data/poisson_gap_mask/'+str(N)+'/Mask_poisson_'+str(np.random.randint(8000, 9999))+'.npy')  # TODO
            
        idx_ones = np.where(Mask==1)
        U = Mask
        U1 = np.random.random()*np.ones([N, N]) + np.random.random([N, N])/5 -0.1
        U1[idx_ones]= 1

        NUS_FID = np.multiply(U, xx)
        NOISE_FID = np.multiply(U1, xx)
        NOISE_input = np.fft.fft2(NOISE_FID)
        # mul = np.max(np.abs(NOISE_input)) * 2
        # mul = 500
        # net_input = np.concatenate([NOISE_input[np.newaxis], NUS_FID[np.newaxis]], axis=0)
        # net_input = torch.complex(torch.from_numpy(net_input.real).float(), torch.from_numpy(net_input.imag).float()) / mul
        # net_label = torch.complex(torch.from_numpy(GT_label.real).float(), torch.from_numpy(GT_label.imag).float()) / mul
        # return  net_input, net_label
        return NUS_FID, NOISE_input, GT_label
             
def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    num_samples = dataset_opt['data_len']
    dataset = NUSNMRDataset(
                num_samples=num_samples,
                split=phase,
                need_LR=(mode == 'LRHR')
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset