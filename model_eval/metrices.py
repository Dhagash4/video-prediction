from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from torchvision import transforms
import lpips

from utils.utils import *

def get_mae(im1, im2):
    
    im1 = im1.cpu().numpy()
    im2 = im2.cpu().numpy()
    
    return mae(im1,im2)

def get_mse(im1, im2):
    
    im1 = im1.cpu().numpy()
    im2 = im2.cpu().numpy()
    
    return mse(im1,im2)

def get_ssim(im1, im2):
    
    im1 = im1.cpu().numpy()
    im2 = im2.cpu().numpy()
    
    return ssim(im1, im2)

def get_psnr(im1, im2):
    
    im1 = im1.cpu().numpy()
    im2 = im2.cpu().numpy()
    
    return psnr(im1, im2)

def calculate_metrices(gt_seq, pred_seq, device,loss_fn_vgg,batch_first = False):
    
    
    if batch_first:
        gt_seq = gt_seq.permute(1,0,2,3,4)
        pred_seq = pred_seq.permute(1,0,2,3,4)
    
    if gt_seq.shape[0] != pred_seq.shape[0]:
        print("Ground truth sequences (gt_seq) and predicted sequences(pred_Seq) are of diff length")
    
    seq_len, batch_size, n_channels, _, _ = gt_seq.shape

    lpips_seq =  np.zeros((batch_size, seq_len))
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    for i in range(batch_size):
        normalized_gt_seq = transform(gt_seq[:,i,:].expand(-1,3,-1,-1)).to(device)
        normalized_pred_Seq = transform(pred_seq[:,i,:].expand(-1,3,-1,-1)).to(device)
        lpips_dist = loss_fn_vgg(normalized_gt_seq, normalized_pred_Seq) 
        lpips_dist = lpips_dist.data.cpu().numpy()
        lpips_dist = lpips_dist.reshape((-1,))
        lpips_seq[i] = lpips_dist
    
    ssim_seq = np.zeros((batch_size, seq_len))
    psnr_seq = np.zeros((batch_size, seq_len))
    mse_seq = np.zeros((batch_size, seq_len))
    mae_seq = np.zeros((batch_size, seq_len))
    for i in range(batch_size):
        for j in range(seq_len):
            for c in range(n_channels):
                ssim_seq[i,j] += get_ssim(gt_seq[j,i,c], pred_seq[j,i,c])
                psnr_seq[i,j] += get_psnr(gt_seq[j,i,c], pred_seq[j,i,c])
                mse_seq[i,j] += get_mse(gt_seq[j,i,c], pred_seq[j,i,c])
                mae_seq[i,j] += get_mae(gt_seq[j,i,c], pred_seq[j,i,c])
            ssim_seq[i,j] /= n_channels
            psnr_seq[i,j] /= n_channels
            mse_seq[i,j] /= n_channels
            mae_seq[i,j] /= n_channels
    
    return lpips_seq, ssim_seq, psnr_seq, mse_seq, mae_seq