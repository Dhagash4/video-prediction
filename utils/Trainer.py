import torch
import torch.nn as nn
from tqdm import tqdm
from utils.visualizations import show_grid
from utils.utils import get_data_batch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import lpips

class TrainerBase:
    """
    Class for initializing network and training it
    """
    def __init__(self,  config, device, encoder, decoder, predictor,optimizer,scheduler,save_path,writer, resume_point):
        
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
        self.loss_type = self.cfg['train']['loss']
        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.writer = SummaryWriter(writer)
        self.save_path = save_path
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.beta1 = self.cfg['train']['beta1']
        self.best_val_loss = 1e4
        
        """transfer to gpu"""

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.predictor = predictor.to(device)


        """Optimization Parameters"""
        
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.predictor.parameters())
        self.model_optimizer = self.optimizer(params,lr=self.lr, betas = (0.9, 0.999))

        # self.model_scheduler = self.scheduler(self.model_optimizer,step_size = self.cfg['train']['step_size'], gamma=self.cfg['train']['gamma'])
        # self.model_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer,verbose=True)
        self.model_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer,gamma=0.9,verbose=True)

        if self.loss_type == "mse":
            self.loss = nn.MSELoss()
        elif self.loss_type == "lpips":
            self.loss = lpips.LPIPS(net="vgg").to(device)

    def train_one_step(self, x):
        
        """ Training all models for one optimization step """
        self.model_optimizer.zero_grad()
        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        # self.predictor_optimizer.zero_grad()

        self.predictor.init_hidden_states()
        loss = 0
        
    
        for i in range(0,(self.past_frames+self.future_frames)-1):
            
            encoded = self.encoder(x[i])
            lstm_outputs = self.predictor(encoded)
        
            x_pred = self.decoder([encoded,lstm_outputs])
            # x_pred = self.decoder(lstm_outputs)
    
            # h_prev = self.encoder(x[i])
            if self.loss_type == "mse":
                loss += self.loss(x_pred,x[i+1])

            elif self.loss_type == "lpips":
                normalized_gt_seq = self.transform(x[i+1].expand(-1,3,-1,-1))
                normalized_pred_seq = self.transform(x_pred.expand(-1,3,-1,-1))
                loss += torch.mean(self.loss.forward(normalized_pred_seq,normalized_gt_seq))
       
        loss.backward()

        self.model_optimizer.step()
        

        return loss.data.cpu().numpy()/(((self.past_frames+self.future_frames)))
    
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
                x_input = self.decoder([encoded_skips,lstm_outputs])
                # x_input = self.decoder(lstm_outputs)
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
        val_loss = 0.0
        x_in = x[0]
        
        for i in range(1,self.past_frames+self.future_frames):
            
            encoded_skips = self.encoder(x_in)
                
            if i < self.past_frames:
                
                self.predictor(encoded_skips)
                x_in = x[i]
            
            else:
                
                lstm_outputs = self.predictor(encoded_skips)
                x_in = self.decoder([encoded_skips,lstm_outputs])
                if self.loss_type == "mse":
                    val_loss += self.loss(x_in,x[i])
                
                elif self.loss_type == "lpips":
                    normalized_gt_seq = self.transform(x[i].expand(-1,3,-1,-1))
                    normalized_pred_seq = self.transform(x_in.expand(-1,3,-1,-1))
                    val_loss += torch.mean(self.loss.forward(normalized_pred_seq,normalized_gt_seq))
        
       
        return val_loss.data.cpu().numpy() / ((self.future_frames))

    def train(self, train_loader,val_loader,test_loader):
        
        """ Training the models for several iterations """
        
        niter = 0
        val_iter = 0
        num_epochs = self.cfg['train']['max_epoch']

        training_batch_generator = get_data_batch(self.cfg,train_loader)
        test_batch = next(iter(test_loader))
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
                progress_bar.set_description(f"Epoch {i+1} Iter {niter+1}: loss {epoch_mse:.5f} ")
                self.writer.add_scalar(f'Training Loss',mse, global_step=niter)
                niter+=1

            """validation loop"""

            val_progress = tqdm(enumerate(val_loader), total=len(val_loader))

            for _ , seqs in val_progress:

                seqs = seqs.to(self.device)
                loss = self.eval_one_step(seqs)
                val_loss+=loss
                val_progress.set_description(f"Val Epoch {i+1} Iter {val_iter+1}: val_loss {val_loss:.5f} ")
                self.writer.add_scalar(f'Validation Loss',loss, global_step = val_iter)
                val_iter+=1
            
            
            if self.cfg['data']['dataset'] == 'KTH':
                self.writer.add_scalar(f'Training Loss per epoch', epoch_mse / (len(train_loader)), global_step=i)
                print(f"Training Loss:\ntrain_loss: {epoch_mse/ len(train_loader)}, val_loss: {val_loss / len(val_loader)}")
            else:
                self.writer.add_scalar(f'Training Loss per epoch', epoch_mse / (self.cfg['data']['niters']), global_step=i)
                print(f"Training Loss:\ntrain_loss: {epoch_mse/ (self.cfg['data']['niters'])}, val_loss: {val_loss / len(val_loader)}")
            
            self.writer.add_scalar(f'Validation Loss per epoch', val_loss/ (len(val_loader)), global_step = i)
            
        
            all_gen = self.generate_future_sequences(test_batch)
            grid = show_grid(test_batch,all_gen,nsamples=5,pred_frames=self.past_frames)
            self.writer.add_image('images', grid, global_step=i)

            """scheduler step"""

            if (i % self.cfg['train']['step_size'] == 0) and (i != 0) :
                self.model_scheduler.step()
            

            """saving entities"""

            if self.resume_point != 0 and (((i % self.cfg['train']['model_save']) == 0)):

                torch.save({
                        'encoder': self.encoder,
                        'decoder':self.decoder,
                        'predictor': self.predictor,
                        'config': self.cfg},
                        f'{self.save_path}/model_{i+self.resume_point+1}.pth')
            elif ((i % self.cfg['train']['model_save']) == 0):

                torch.save({
                            'encoder': self.encoder,
                            'decoder':self.decoder,
                            'predictor': self.predictor,
                            'config': self.cfg},
                            f'{self.save_path}/model_{i}.pth')


        """saving last model"""
        torch.save({    'encoder': self.encoder,
                        'decoder':self.decoder,
                        'predictor': self.predictor,
                        'config': self.cfg},
                        f'{self.save_path}/model_{i}.pth')