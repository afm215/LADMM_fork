import torch
import torch.nn as nn
import tqdm 
from models.pnp_blocks import upsample, o_leary_batch, transpose_o_leary_batch
from models.pnp_blocks import ResUNet as net
import time
import numpy as np


class PnP_ALPHA_PGD():
    def __init__(self, opt):
        self.opt = opt
        self.device = 'cuda'
        self.gamma = self.opt['gamma']
        self.lamb = self.opt['lamb']
        self.alpha = self.opt['alpha']
        self.sigma_d = torch.FloatTensor([[[[self.opt['sigma_d']]]]]).to(self.device)
        self.n_iter = self.opt['n_iter']
        self.load_denoiser()

    def get_hyperparams(self):
        return self.gamma, self.sigma_d.cpu().item(), None

    def load_denoiser(self):
        self.denoiser = net(in_nc=4, out_nc=3,   nc=[64, 128, 256, 512],
                            nb=4, act_mode='R',  downsample_mode='strideconv',
                            upsample_mode='convtranspose')
        self.denoiser.load_state_dict(torch.load(self.opt['path_denoiser']))
        self.denoiser = self.denoiser.to(self.device)

    def feed_data(self, data):
        self.y = data['L'].to(self.device)
        self.sf = data['sf'][0]
        self.sigma = data['sigma'].to(self.device)
        self.kmap = data['kmap'].to(self.device)
        self.basis = data['basis'].to(self.device)
        try:
            self.ref = data['H']
        except:
            pass


    def grad_data_term(self, x, kmap, basis):
        res = o_leary_batch(x, kmap, basis) - self.y
        return transpose_o_leary_batch(res, kmap, basis)
    
    def grad_reg_term(self,x, sigma):
        x = x.float()
        x = x.requires_grad_()

        N = self.denoiser(x, sigma)
       
        JN = torch.autograd.grad(N, x, grad_outputs=x - N, create_graph=True, only_inputs=True)[0]
        Dg = x - N - JN
        return Dg, N
        
    def run_iter(self, x, y_k, sigma, kmap, basis):
        # x = x - self.gamma * self.grad_data_term(x, y, sigma, kmap, basis)
        # x = self.denoiser(torch.cat((x, self.sigma_d.repeat(x.shape[0],1,x.shape[2], x.shape[3])), dim=1))


        q = ( 1 - self.alpha) * y_k + self.alpha * x
        gradx = self.grad_data_term(q, kmap, basis)
        z = x - self.lamb*gradx # gradx is gradfx # tau is 1 mandatory?
        # Denoising step
        torch.set_grad_enabled(True)
        Dg, N = self.grad_reg_term(z, self.sigma_d / 255.)
        torch.set_grad_enabled(False)
        Dg = Dg.detach()
        N = N.detach()
        g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
        Dz = z - self.gamma * Dg
        Dx = Dz
        y_k = (1 - self.alpha)*(y_k) + self.alpha*Dz
        x = Dz

        return x, y
    
    def init_pnp(self, y, sf):
        upsampler = nn.Upsample(scale_factor=sf, mode='bilinear', align_corners=False)
        x = upsampler(y).to(self.device)
        
        return x
    
    def run(self):
        with torch.no_grad():
            x = self.init_pnp(self.y, self.sf)
            x_list = [x]
            time_list = [0]
            y_k = x
            for _ in range(self.n_iter):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                x, y_k = self.run_iter(x, y_k, self.sigma, self.kmap, self.basis)
                torch.cuda.synchronize()
                t2 = time.perf_counter()

                x_list.append(x.cpu())
                time_list.append(t2-t1)
            
        return x.cpu(), x_list, time_list


    # def fit_params(self):
    #     gamma_grid = [0.1, 0.5, 0.9, 1.5, 2]
    #     sig_grid = [5/255, 10/255, 20/255, 40/255, 60/255]
    #     current_mse = np.inf
    #     current_params = (None,None)

    #     # Grid search
    #     for sig in tqdm.tqdm(sig_grid, desc='Sigma loop'):
    #         sigD = torch.FloatTensor([[[[sig]]]]).to(self.device)
    #         self.sigma_d = sigD
    #         for gamma in tqdm.tqdm(gamma_grid, desc='Gamma loop'):
    #             self.gamma = gamma

    #             est, _, _ = self.run()
    #             mse = ((est - self.ref)**2)[...,17:-17,17:-17].mean().item()
    #             #mse = 0

    #             if mse <= current_mse:
    #                 current_mse = mse
    #                 current_params = (sigD, gamma)

    #     # Update model with optimal params
    #     self.sigma_d  = current_params[0]
    #     self.gamma = current_params[1]

    #     print(60 * '-')
    #     print('Grid search completed!')
    #     print(60 * '-')
