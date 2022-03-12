from models.vgg_baseline import *
from models.convlstm_baseline import *
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.visualizations import *
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
import shutil

class TrainerFP:
    """
    Class for initializing network and training it
    """
    def __init__(self, batch_size =40, in_size = (1,64,64), device="cpu", writer=None,save_path = None):
        """ Initialzer """
        assert writer is not None, f"Tensorboard writer not set..."
        assert save_path is not None, f"Checkpoint saving directory not set..."
        
        self.past_frames = 10
        self.future_frames = 10 
       
        self.lr = 0.002
        self.beta = 0.0001
        
        self.batch_size = batch_size
        self.last_frame_skip = True
        self.device = device
        self.save_path = save_path
        self.writer = SummaryWriter(writer)

        # if train_type == "Fixed Prior" and img_size==64:
        self.encoder = VGG_Encoder()
        self.decoder = VGG_Decoder()
        self.lstm  = LSTM_baseline()
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.lstm = self.lstm.to(device)


        # Decay LR by a factor of 0.1 every 5 epochs
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas = (0.9, 0.999))

        self.mse_loss = nn.MSELoss()
    
    def kld_loss(self, mu, log_var):
        """
        Combined loss function for joint optimization of 
        prediction and ELBO
        """
        # mse = F.mse_loss(predicted, real, reduction="sum")
    #     recons_loss = F.binary_cross_entropy(recons.view(b_size,-1), target.view(b_size,-1), reduction='sum')
        elbo = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        elbo /= self.batch_size
        # total_loss = mse + elbo*beta
        return elbo

    def train_one_step(self, x):
        """ Training both models for one optimization step """
        
        
        self.lstm.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        self.predictor.h, self.predictor.c = self.predictor.init_state(b_size=self.batch_size, device=self.device)
        self.posterior.h, self.posterior.c = self.posterior.init_state(b_size=self.batch_size, device=self.device)
        # print("hello")

        h_seq = [self.encoder(x[i]) for i in range(self.past_frames+self.future_frames)]
        # print(h_seq[0][0].shape)
        mse =0
        kld = 0
        for i in range(1,self.past_frames+self.future_frames):
            h_target = h_seq[i][0]
            if self.last_frame_skip or i<self.past_frames:
                h, skip = h_seq[i-1]
            else:
                h = h_seq[i-1][0]
            # print("here")
            # print(h_target.shape)
            z_t, mu, log_var = self.posterior(h_target)
            # print("here")
            # print(z_t.shape)
            h_pred = self.predictor(torch.cat([h, z_t], 1))
            x_pred = self.decoder([h_pred, skip])
            mse+=self.mse_loss(x_pred,x[i])
            kld+=self.kld_loss(mu,log_var)

        total_loss = mse + kld*self.beta
        total_loss.backward()

        self.predictor_optimizer.step()
        self.posterior_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return mse.data.cpu().numpy()/(self.past_frames+self.future_frames), kld.data.cpu().numpy()/(self.past_frames+self.future_frames)
    
    @torch.no_grad()
    def generate_future_sequences(self, test_batch):
        """ Generating a bunch of images using current state of generator """
        self.predictor.eval()
        self.posterior.eval()
        self.encoder.eval()
        self.decoder.eval()

        all_gen = []
        gt_seq = []
        pred_seq = []
        x_in = test_batch[0]
        all_gen.append(x_in)
        for i in range(1, self.past_frames+self.future_frames):
            h = self.encoder(x_in)
            if self.last_frame_skip or i < self.past_frames:
                h, skip = h
            else:
                h, _ = h
            h = h.detach()

            if i< self.past_frames:
                h_target = self.encoder(test_batch[i])[0].detach()
                z_t, _, _ = self.posterior(h_target)
            else:
                z_t = torch.cuda.FloatTensor(self.batch_size, self.z_dim).normal_()
            
            if i < self.past_frames:
                self.predictor(torch.cat([h, z_t], 1))
                x_in = test_batch[i]
                all_gen.append(x_in)
            else:
                h = self.predictor(torch.cat([h, z_t], 1)).detach()
                x_in = self.decoder([h, skip]).detach()
                pred_seq.append(x_in)
                gt_seq.append(test_batch[i])
                all_gen.append(x_in) 
        
        # gt_seq.extend(pred_seq)
        # predicted_batch = torch.stack(gt_seq)

        all_gen = torch.stack(all_gen)
        gt_seq = torch.stack(gt_seq)
        pred_seq = torch.stack(pred_seq)

        return all_gen, gt_seq, pred_seq

    def get_training_batch(self,train_loader,dtype=torch.cuda.FloatTensor):
        while True:
            for sequence in train_loader:
                batch = utils.normalize_data(dtype, sequence)
                batch = torch.stack(batch)
                yield batch

    def get_testing_batch(self,test_loader,dtype=torch.cuda.FloatTensor):
        while True:
            for sequence in test_loader:
                batch = utils.normalize_data(dtype, sequence)
                batch = torch.stack(batch)
                yield batch 

    def train(self, train_loader, val_loader, test_loader, num_epochs=300, device= "cpu", init_step=0):
        """ Training the models for several iterations """
        
        niter = 0
        training_batch_generator = self.get_training_batch(train_loader)
        testing_batch_generator = self.get_testing_batch(val_loader)
        test_batch = next(testing_batch_generator)
        
        # test_batch = next(iter(val_loader))
        # test_batch = test_batch.to(device)
        

        for i in range(num_epochs):
            self.predictor.train()
            self.posterior.train()
            self.encoder.train()
            self.decoder.train() 
            epoch_mse =0
            epoch_kld=0
            epoch_loss =0

            progress_bar = tqdm(range(600), total=600)
            for j in progress_bar:
                seqs = next(training_batch_generator)
            # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            # for _,seqs in progress_bar:
                seqs = seqs.to(device)
                mse, kld = self.train_one_step(seqs)
                epoch_mse+=mse
                epoch_kld+=kld
                progress_bar.set_description(f"Epoch {i+1} Iter {niter+1}: mse loss {epoch_mse:.5f}, kld los {epoch_kld: .5f}")
                
                # write mse and kld losses to tensorboard after every iteration
                self.writer.add_scalar(f'MSE Loss',mse, global_step=niter)
                self.writer.add_scalar(f'KLD Loss',kld, global_step=niter)
                self.writer.add_scalars(f'Comb_Loss/Losses', {'MSE':mse,'KLD': kld}, niter)    
                niter+=1


            #after every epoch calculate losses for validation and training datasets
            #to add validation step

            print(f"Training Loss:\nMSE: {epoch_mse}, KLD: {epoch_kld}\n")
            

            #save models and gifs after every 10 epoch
            
            if(i%20==0) or (i == num_epochs-1):

                save_gif_batch(test_batch, nsamples=5, text = "real", show=False)
                all_gen, gt_seq, pred_seq = self.generate_future_sequences(test_batch)
                grid = show_grid(test_batch,all_gen,nsamples=5,pred_frames=self.past_frames)
                self.writer.add_image('images', grid, global_step=niter)
                save_pred_gifs(pred_seq, nsamples = 5, text=f"epoch_{i+1}",show=False)

                torch.save({
                    'encoder': self.encoder,
                    'decoder':self.decoder,
                    'predictor': self.predictor,
                    'posterior': self.posterior},
                    f'{self.save_path}/model_{i}.pth')
                
            self.writer.close()