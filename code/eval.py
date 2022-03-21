import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from model_eval.metrices import calculate_fvd, calculate_metrices
from utils.utils import eval_dataset, set_random_seed
from utils.visualizations import save_grid_batch


class eval():

    def __init__(self, model_path, batch_size = 16, past_frames=10, future_frames=10) -> None:

        self.batch_size = batch_size
        self.past_frames = past_frames
        self.future_frames = future_frames    
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        saved_model = torch.load(model_path,map_location=self.device)

        self.encoder = saved_model['encoder']
        self.decoder = saved_model['decoder']
        self.predictor = saved_model['predictor']
        self.cfg = saved_model['config']
        skip = self.cfg['architecture']['skip']
        dataset = self.cfg['data']['dataset']
        self.predictor.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.predictor.convlstm1.batch_size = self.batch_size
        self.predictor.convlstm2.batch_size = self.batch_size
        self.predictor.convlstm3.batch_size = self.batch_size


        """loading datasets"""

        test_loader = eval_dataset(dataset=dataset,batch_size=self.batch_size)
        self.len_test_loader = len(test_loader)
        """setting random seed"""

        set_random_seed(random_seed=3292666)

        self.eval_sequences = []

        ssim = []
        lpips_ = []
        psnr= []
        mse = []
        mae = []
        fvd_ = []
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)

        for i,seqs in tqdm(enumerate(test_loader),total =len(test_loader)):
            
            seqs = seqs.to(self.device)

            gt_seq,pred_seq = self.generate_future_sequences(seqs,skip=skip)
            lpips_seq, ssim_seq, psnr_seq, mse_seq, mae_seq = calculate_metrices(gt_seq,pred_seq,self.device,loss_fn_vgg=loss_fn_vgg)
            lpips_.append(lpips_seq)
            ssim.append(ssim_seq)
            psnr.append(psnr_seq)
            mse.append(mse_seq)
            mae.append(mae_seq)
            fvd_.apend(calculate_fvd(gt_seq,pred_seq))
            self.eval_sequences.append([seqs.cpu().numpy(),np.concatenate((gt_seq.cpu().numpy(),pred_seq.cpu().numpy()),axis=0)])

        # eval_sequences = torch.stack(eval_seq)
        self.lpips_ = np.concatenate((lpips_))
        self.ssim   = np.concatenate((ssim))
        self.mse    = np.concatenate((mse))
        self.mae    = np.concatenate((mae))
        self.psnr   = np.concatenate((psnr))
        self.fvd_   = np.array(fvd_)

    @torch.no_grad()
    def generate_future_sequences(self, test_batch,skip=False):

        """ Generating a bunch of images using current state of generator """
        
        self.predictor.init_hidden_states()
        gt_seq = []
        pred_seq = []
        x_input = test_batch[0]
        for i in range(1, self.past_frames+self.future_frames):
            
            encoded_skips = self.ncoder(x_input)
                
            if i < self.past_frames:
                self.predictor(encoded_skips)
                x_input = test_batch[i]
            else:
                lstm_outputs = self.predictor(encoded_skips)
                x_input = self.decoder([encoded_skips,lstm_outputs])
                gt_seq.append(test_batch[i])
                pred_seq.append(x_input)
        
        gt_seq = torch.stack(gt_seq)
        pred_seq = torch.stack(pred_seq)
        
        return gt_seq,pred_seq
    
    def visualize_best_metrices(self):

        best_lpips_index = np.argmax(np.mean(self.lpips_, axis=1))
        best_ssim_index = np.argmax(np.mean(self.ssim, axis=1))
        best_psnr_index = np.argmax(np.mean(self.psnr, axis=1))
        best_mse_index = np.argmin(np.mean(self.mse, axis=1))
        best_mae_index = np.argmin(np.mean(self.mae, axis=1))

        indexes = [best_lpips_index, best_ssim_index, best_psnr_index, best_mse_index, best_mae_index]
        gt = [torch.from_numpy(self.eval_sequences[indx][0]) for indx in indexes]
        gt = torch.cat(gt, dim=1)

        pred = [torch.from_numpy(self.eval_sequences[indx][1]) for indx in indexes]
        pred = torch.cat(pred, dim=1)
        
        save_grid_batch(gt, pred,  text = "best_metrices", show = True)
    
    def visualize_best_fvd_batch(self):

        best_fvd_batch = np.argmin(self.fvd_)
        batch_start_indx = best_fvd_batch * self.batch_size
        batch_end_index = (best_fvd_batch + 1) * self.batch_size
        batch = self.eval_sequences[batch_start_indx  : batch_end_index]

        gt = [torch.from_numpy(self.eval_sequences[indx][0]) for indx in batch]
        gt = torch.cat(gt, dim=1)

        pred = [torch.from_numpy(self.eval_sequences[indx][1]) for indx in batch]
        pred = torch.cat(pred, dim=1) 

        save_grid_batch(gt, pred, nsamples=5, text = "best_fvd", show = True)




