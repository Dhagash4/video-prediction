# from models.dcgan import *
# from models.lstm import *
# from models.resnet import *
from models.baselineLSTM import predictor
from models.vgg_baseline import *
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
    def __init__(self, arch_type = "dcgan", lstm_type = "lstm", batch_size =20, embed_dim=128, hidden_dim=256,
                 img_size=(1,64,64), device="cpu", writer=None,save_path=None):
        """ Initialzer """
        assert writer is not None, f"Tensorboard writer not set..."
        assert save_path is not None, f"Checkpoint saving directory not set..."
        
        self.past_frames = 10
        self.future_frames = 10 
        self.g_dim = embed_dim
        self.img_size = img_size
        self.lr = 0.002
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.last_frame_skip = True
        self.device = device
        self.writer = SummaryWriter(writer)
        self.encoder = VGGEncoder()
        self.decoder = VGGDecoder()
        self.predictor = predictor(device = device)
        self.save_path = save_path
        
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.predictor = self.predictor.to(device)
        # self.predictor = self.predictor.to(device)
        # Decay LR by a factor of 0.1 every 5 epochs
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
        # self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.lr, betas = (0.9, 0.999))
        self.mse_loss = nn.MSELoss()

    def train_one_step(self, x):
        """ Training both models for one optimization step """
        
        # self.predictor_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        self.predictor_optimizer.zero_grad()

        self.predictor.init_hidden_states()
        # self.predictor.h, self.predictor.c = self.predictor.init_state()

        h_prev = self.encoder(x[0])
        # print(h_seq[0][0].shape)
        mse =0
        for i in range(1,self.past_frames+self.future_frames):
            # encoded_skip, lstm_outputs = h_seq[i-1]
            lstm_ouputs = self.predictor(h_prev)
            x_pred = self.decoder(lstm_ouputs)
            h_prev = self.encoder(x[i])
            mse+=self.mse_loss(x_pred,x[i])

        mse.backward()

        self.predictor_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return mse.data.cpu().numpy()/(self.past_frames+self.future_frames)
    
    @torch.no_grad()
    def generate_future_sequences(self, test_batch):
        """ Generating a bunch of images using current state of generator """
        self.predictor.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        self.predictor.init_hidden_states()
        all_gen = []
        gt_seq = []
        pred_seq = []
        x_in = test_batch[0]
        all_gen.append(x_in)
        for i in range(1, self.past_frames+self.future_frames):
            encoded_skips = self.encoder(x_in)
            encoded_skips = [encoded.detach() for encoded in encoded_skips]
            # if self.last_frame_skip or i < self.past_frames:
            #     h, skip = h
            # else:
            #     h, _ = h            
            if i < self.past_frames:
                self.predictor(encoded_skips)
                x_in = test_batch[i]
                all_gen.append(x_in)
            else:
                lstm_outputs = self.predictor(encoded_skips)
                lstm_outputs = [out.detach() for out in lstm_outputs]
                x_in = self.decoder(lstm_outputs).detach()
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
        test_batch = next(iter(val_loader))
        test_batch = test_batch.to(device)
        # save_gif_batch(test_batch, nsamples=1, text = "real", show=False)

        for i in range(num_epochs):
            self.predictor.train()
            self.encoder.train()
            self.decoder.train() 
            epoch_mse =0

            progress_bar = tqdm(range(400), total=400)
            for j in progress_bar:
                seqs = next(training_batch_generator)
                seqs = seqs.to(device)
                mse = self.train_one_step(seqs)
                epoch_mse+=mse
                progress_bar.set_description(f"Epoch {i+1} Iter {niter+1}: mse loss {epoch_mse:.5f} ")
                
                # write mse and kld losses to tensorboard after every iteration
                self.writer.add_scalar(f'MSE Loss',mse, global_step=niter)
                niter+=1


            #after every epoch calculate losses for validation and training datasets
            print(f"Training Loss:\nMSE: {epoch_mse}")

            #save models and gifs after every 10 epoch
            
          

            all_gen, gt_seq, pred_seq = self.generate_future_sequences(test_batch)
            grid = show_grid(test_batch,all_gen,nsamples=5,pred_frames=self.past_frames)
            self.writer.add_image('images', grid, global_step=niter)
            torch.save({
                    'encoder': self.encoder,
                    'decoder':self.decoder,
                    'predictor': self.predictor},
                    f'{self.save_path}/model_{i}.pth')
            # save_pred_gifs(pred_seq, nsamples = 1, text=f"{niter}_epoch_{i+1}",show=False)
            # save_grid_batch(test_batch, all_gen, nsamples = 1, text=f"{niter}_grid_epoch_{i+1}",show=False)

            

