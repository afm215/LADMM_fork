# -*- coding: utf-8 -*-
import torch.fft as fft
import torch

import utils.utils_image as util
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def splits(a, sf):
    '''split a into sfxsf distinct blocks
    Args:
        a: NxCxWxH
        sf: split factor
    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.
    Args:
        psf: NxCxhxw
        shape: [H, W]
    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler
    Upsampling the spatial size by filling the new entries with zeros
    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler
    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others
    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]



def data_solution(x, FB, FBC, F2B, FBFy, alpha, sf):
    FR = FBFy + fft.fftn(alpha*x, dim=(-2,-1))
    x1 = FB.mul(FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = FBR.div(invW + alpha)
    FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
    FX = (FR-FCBinvWBR)/alpha
    Xest = torch.real(fft.ifftn(FX, dim=(-2, -1)))

    return Xest


def pre_calculate(x, k, sf):
    '''
    Args:
        x: NxCxHxW, LR input
        k: NxCxhxw
        sf: integer

    Returns:
        FB, FBC, F2B, FBFy
        will be reused during iterations
    '''
    w, h = x.shape[-2:]
    FB = p2o(k, (w*sf, h*sf))
    FBC = torch.conj(FB)
    F2B = torch.pow(torch.abs(FB), 2)
    STy = upsample(x, sf=sf)
    FBFy = FBC*fft.fftn(STy, dim=(-2, -1))
    return FB, FBC, F2B, FBFy


def get_metrics(est_list, ref, border=17):
    psnr_list, ssim_list, lpips_list = [], [], []
    psnr = PeakSignalNoiseRatio(data_range=1)
    ssim = StructuralSimilarityIndexMeasure()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    #lpips.net = lpips.net.to('cuda')
    with torch.no_grad():
        ref = ref.cpu()[...,border:-border,border:-border]
        for est in est_list:
            est = est.cpu().clamp(0,1)[...,border:-border,border:-border]
            psnr_list.append(psnr(ref, est).cpu().item())
            ssim_list.append(ssim(ref, est).cpu().item())
            lpips_list.append(lpips(ref,est).cpu().item())
        
    return psnr_list, ssim_list, lpips_list

def get_metrics_bis(est_list, ref, border=17):
    psnr_list, ssim_list, lpips_list = [], [], []
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    ref = ref.cpu()[...,border:-border,border:-border]
    ref_batch = [util.tensor2uint(i) for i in ref]
    with torch.no_grad():
        for est in est_list:
            est = est.cpu().clamp(0,1)[...,border:-border,border:-border]
            est_batch = [util.tensor2uint(i) for i in est]
            for i,j in zip(est_batch, ref_batch):
                psnr_list.append(peak_signal_noise_ratio(j,i))
                ssim_list.append(structural_similarity(j,i, data_range=1, multichannel=True))
            lpips_list.append(lpips(ref,est).cpu().item())
        
    return psnr_list, ssim_list, lpips_list
