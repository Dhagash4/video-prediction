# from models.dcgan import *
# from models.lstm import *
# from models.resnet import *

from unittest import skip
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.visualizations import *
import utils.utils as utils
from torch.utils.tensorboard import SummaryWriter

class TrainerBase:
    """
    Class for initializing network and training it
    """
    def __init__(self,  config, device, encoder, decoder, predictor,optimizer,save_path,writer, resume_point):
        
        """ Initialzer """
        
        self.cfg = config
        self.past_frames = self.cfg['data']['seed_frames']
        self.future_frames = self.cfg['data']['predict_frames']
        self.lr = self.cfg['train']['lr']
        self.batch_size = self.cfg['train']['batch_size']
        self.last_frame_skip = True
        self.device = device
        self.resume_point = resume_point
        self.skip_connection = self.cfg['architecture']['skip']
       
        self.writer = SummaryWriter(writer)
        self.save_path = save_path
        
        self.optimizer = optimizer
        self.beta1 = self.cfg['train']['beta1']
        self.best_val_loss = 1e4
        
        """transfer to gpu"""

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.predictor = predictor.to(device)


        """Optimization Parameters"""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.predictor.parameters())
        self.model_optimizer = self.optimizer(params,lr=self.lr, betas = (0.9, 0.999))
        # self.encoder_optimizer = self.optimizer(self.encoder.parameters(), lr=self.lr, betas = (0.9, 0.999))
        # self.decoder_optimizer = self.optimizer(self.decoder.parameters(), lr=self.lr, betas = (0.9, 0.999))
        # self.predictor_optimizer = self.optimizer(self.predictor.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.loss = nn.MSELoss()

    def train_one_step(self, x):
        
        """ Training all models for one optimization step """
        self.model_optimizer.zero_grad()
        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        # self.predictor_optimizer.zero_grad()

        self.predictor.init_hidden_states()
        mse = 0
        
    
        for i in range(0,(self.past_frames+self.future_frames)-1):
            
            encoded = self.encoder(x[i])
            lstm_outputs = self.predictor(encoded)
        
            x_pred = self.decoder([encoded,lstm_outputs],skip_connection= self.skip_connection)
    
            # h_prev = self.encoder(x[i])
            mse += self.loss(x_pred,x[i+1])
       
        mse.backward()

        self.model_optimizer.step()

        return mse.data.cpu().numpy()/(self.past_frames+self.future_frames)
    
    @torch.no_grad()
    def generate_future_sequences(self, test_batch):

        """ Generating a bunch of images using current state of generator """
        
        self.predictor.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        self.predictor.init_hidden_states()
        all_gen = []
        x_input = test_batch[0]
        all_gen.append(x_input)
        
        for i in range(1, self.past_frames+self.future_frames):
            
            encoded_skips = self.encoder(x_input)
                
            if i < self.past_frames:
                self.predictor(encoded_skips)
                x_input = test_batch[i]
                all_gen.append(x_input)
            else:
                lstm_outputs = self.predictor(encoded_skips)
                x_input = self.decoder([encoded_skips,lstm_outputs],skip_connection= self.skip_connection)
                all_gen.append(x_input) 
        
       
        all_gen = torch.stack(all_gen)
        
        return all_gen

    
    @torch.no_grad()
    def eval_one_step(self, x):
        
        """ Generating a bunch of images using current state of generator """
        
        self.predictor.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.predictor.init_hidden_states()
        val_mse = 0.0
        x_in = x[0]
        
        for i in range(1,self.past_frames+self.future_frames):
            
            encoded_skips = self.encoder(x_in)
                
            if i < self.past_frames:
                
                self.predictor(encoded_skips)
                x_in = x[i]
            
            else:
                
                lstm_outputs = self.predictor(encoded_skips)
                
                x_in = self.decoder([encoded_skips,lstm_outputs],skip_connection= self.skip_connection)

                val_mse+=self.loss(x_in,x[i])
        
       
        return val_mse.data.cpu().numpy()/(self.future_frames)

    def train(self, train_loader,val_loader,test_loader):
        
        """ Training the models for several iterations """
        
        niter = 0
        val_iter = 0
        num_epochs = self.cfg['train']['max_epoch']

        training_batch_generator = utils.get_data_batch(self.cfg,train_loader)
        test_batch_generator = utils.get_data_batch(self.cfg,test_loader)
        test_batch = next(iter(test_batch_generator))
        test_batch = test_batch.to(self.device)


        for i in range(num_epochs):
           
            """Setting to train"""

            self.predictor.train()
            self.encoder.train()
            self.decoder.train() 

            """training loop"""

            epoch_mse =0
            val_loss = 0
            if self.cfg['data']['dataset'] == 'KTH':

                progress_bar = tqdm(range(len(train_loader)), total = len(train_loader))

            else:

                progress_bar = tqdm(range(self.cfg['data']['niters']), total=self.cfg['data']['niters'])
            
            for j in progress_bar:
                
                seqs = next(training_batch_generator)
                seqs = seqs.to(self.device)
                mse = self.train_one_step(seqs)
                epoch_mse+=mse
                progress_bar.set_description(f"Epoch {i+1} Iter {niter+1}: mse_loss {epoch_mse:.5f} ")
                self.writer.add_scalar(f'Training Loss',mse, global_step=niter)
                niter+=1

            # """validation loop"""

            val_progress = tqdm(enumerate(val_loader), total=len(val_loader))

            for _ , seqs in val_progress:

                seqs = seqs.to(self.device)
                loss = self.eval_one_step(seqs)
                val_loss+=loss
                val_progress.set_description(f"Val Epoch {i+1} Iter {val_iter+1}: val_loss {val_loss:.5f} ")
                val_iter+=1

            self.writer.add_scalar(f'Validation Loss',val_loss / len(val_loader), global_step=i)
            print(f"Training Loss:\ntrain_loss: {epoch_mse}, val_loss: {val_loss}")
            


            all_gen = self.generate_future_sequences(test_batch)
            grid = show_grid(test_batch,all_gen,nsamples=5,pred_frames=self.past_frames)
            self.writer.add_image('images', grid, global_step=i)

            if self.resume_point != 0:

                torch.save({
                        'encoder': self.encoder,
                        'decoder':self.decoder,
                        'predictor': self.predictor,
                        'config': self.cfg},
                        f'{self.save_path}/model_{i+self.resume_point+1}.pth')
            else:
                torch.save({
                            'encoder': self.encoder,
                            'decoder':self.decoder,
                            'predictor': self.predictor,
                            'config': self.cfg},
                            f'{self.save_path}/model_{i}.pth')

            if val_loss < self.best_val_loss:

                torch.save({
                    'encoder': self.encoder,
                    'decoder':self.decoder,
                    'predictor': self.predictor,
                    'config': self.cfg},
                    f'{self.save_path}/best_model.pth')
 

            

