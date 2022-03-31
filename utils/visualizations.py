from random import sample
import numpy as np
import io
import os
import imageio
import matplotlib.pyplot as plt
from IPython.display import display
import torch
import torchvision
from ipywidgets import widgets, HBox, VBox, Layout, Box


def save_pred_gifs(pred_batch, nsamples=5, text = None, batch_first = False, show = False):

    if batch_first:
        pred_batch = pred_batch.permute(1,0,2,3,4)

    pred_batch = pred_batch.cpu().numpy() * 255.0  
    pred_batch = pred_batch.squeeze(2)

    temp_path = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    # box_layout = Layout(flex_flow='row')
    for i in range(nsamples):
        video = pred_batch[:,i,:]
        # imageio.mimsave(os.path.join(temp_path, f"full_frames{i+1}.gif"),video.astype(np.uint8),"GIF",fps=5)
        imageio.mimsave(os.path.join(temp_path, f"predicted_frames{i+1}_{text}.gif"),video.astype(np.uint8),"GIF",fps=5)
    
    if show:
        pred =[] 
        for i in range(nsamples):
            # full.append(widgets.Image(value=open(os.path.join(temp_path, f"full_frames{i+1}.gif"), 'rb').read()))
            pred.append(widgets.Image(value=open(os.path.join(temp_path, f"predicted_frames{i+1}_{text}.gif"), 'rb').read()))

        print("------------predicted Frames (predicted 10 frames of the sequence)----------------")
        display(HBox(pred))

def save_grid_batch(real_batch, pred_batch=None, nsamples = 5, text = None, batch_first = False, show = False):  
    '''
    sample_batch: batch of sequences shape:(seq_len, batch_size, nc, h, w)
    nsamples: #sequences to display
    '''
    if batch_first:
        real_batch = real_batch.permute(1,0,2,3,4)

    real_batch = real_batch.cpu().numpy() * 255.0 

    if pred_batch is not None:
        if batch_first:
            pred_batch = pred_batch.permute(1,0,2,3,4)

        pred_batch = pred_batch.cpu().numpy() * 255.0  
    seq_len = real_batch.shape[0]

    temp_path = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    if pred_batch is not None:
        fig, ax = plt.subplots(2*nsamples, seq_len)
        fig.set_size_inches(18, 10)
        for i in range(nsamples):
            video = pred_batch[:,i,:]
            real_video = real_batch[:,i,:]
            for j in range(seq_len):
                ax[2*i,j].imshow(real_video[j][0], cmap="gray")
                ax[2*i,j].axis("off")
                
                if j>=10: 
                    ax[2*i+1,j].imshow(video[j][0], cmap="gray")
                    ax[2*i+1,j].axis("off")
                else:
                    h, w =video[j][0].shape
                    whiteFrame = 255 * np.ones((h,w), np.uint8)
                    ax[2*i+1,j].imshow(whiteFrame, cmap="gray")
                    ax[2*i+1,j].axis("off")
        plt.tight_layout()
        # plt.show() 
    else:
        fig, ax = plt.subplots(nsamples, seq_len)
        fig.set_size_inches(18, 5)   
        for i in range(nsamples):
            video = real_batch[:,i,:]
            for j in range(seq_len):
                ax[i,j].imshow(video[j][0], cmap="gray")
                ax[i,j].axis("off")
        plt.tight_layout()
        # plt.show()

    if show:
        plt.savefig(os.path.join(temp_path, f"{text}.png"))
        plt.show() 
        plt.close(fig) 
    else:
        plt.savefig(os.path.join(temp_path, f"{text}.png"))
        plt.close(fig) 


def show_grid(real_batch, pred_batch=None, nsamples = 5, batch_first = False,pred_frames=10):  
    '''
    sample_batch: batch of sequences shape:(seq_len, batch_size, nc, h, w)
    nsamples: #sequences to display
    '''

    assert pred_batch is not None
    sl,b,c,h,w = pred_batch.shape
    if batch_first:

        real_batch = real_batch.permute(1,0,2,3,4)
        pred_batch = pred_batch.permute(1,0,2,3,4)

    pred_batch[:pred_frames] = torch.ones(pred_frames,b,c,h,w)
    
    show_imgs = []
    for i in range(nsamples):
        show_imgs.append(real_batch[:,i,:])
        show_imgs.append(pred_batch[:,i,:])
    
    show_imgs = torch.stack(show_imgs).reshape(-1,c,h,w)

    grid = torchvision.utils.make_grid(show_imgs,nrow=sl,padding=4)
    
    return grid 

def save_gif_batch(real_batch, pred_batch=None,  nsamples = 5, text=None, batch_first = False, show=False):
    '''
    sample_batch: batch of sequences shape:(seq_len, batch_size, nc, h, w)
    nsamples: #sequences to display
    '''
    if batch_first:
        real_batch = real_batch.permute(1,0,2,3,4)
    # Reverse process before displaying
    real_batch = real_batch.cpu().numpy() * 255.0    
    real_batch = real_batch.squeeze(2)
    
    if pred_batch is not None:
        if batch_first:
            pred_batch = pred_batch.permute(1,0,2,3,4)

        pred_batch = pred_batch.cpu().numpy() * 255.0
        pred_batch = pred_batch.squeeze(2)

    temp_path = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    # box_layout = Layout(flex_flow='row')
    for i in range(nsamples):
        video = real_batch[:,i,:]
        # imageio.mimsave(os.path.join(temp_path, f"full_frames{i+1}.gif"),video.astype(np.uint8),"GIF",fps=5)
        if pred_batch is None:
            imageio.mimsave(os.path.join(temp_path, f"past_frames{i+1}_{text}.gif"),video[:10].astype(np.uint8),"GIF",fps=5)
        imageio.mimsave(os.path.join(temp_path, f"past_frames{i+1}_{text}.gif"),video[:10].astype(np.uint8),"GIF",fps=5)
        imageio.mimsave(os.path.join(temp_path,f"future_frames{i+1}_{text}.gif"),video[10:].astype(np.uint8),"GIF",fps=5)
        if pred_batch is not None:
            video = pred_batch[:,i,:]
            imageio.mimsave(os.path.join(temp_path,f"predicted_frames{i+1}_{text}.gif"),video[10:].astype(np.uint8),"GIF",fps=5)

    if show:
        pred =[] 
        past = []
        future = []
        for i in range(nsamples):
            # full.append(widgets.Image(value=open(os.path.join(temp_path, f"full_frames{i+1}.gif"), 'rb').read()))
            past.append(widgets.Image(value=open(os.path.join(temp_path, f"past_frames{i+1}_{text}.gif"), 'rb').read()))
            future.append(widgets.Image(value=open(os.path.join(temp_path,f"future_frames{i+1}_{text}.gif"), 'rb').read()))
            if pred_batch is not None:
                pred.append(widgets.Image(value=open(os.path.join(temp_path,f"predicted_frames{i+1}_{text}.gif"), 'rb').read()))
        # imageA = widgets.Image(value=open('real1.gif', 'rb').read())
        # imageB = widgets.Image(value=open('real2.gif', 'rb').read())

        # hbox = HBox([imageA, imageB])
        # display(hbox)

        print("------------Past Frames (First 10 frames of the sequence)----------------")
        display(HBox(past))

        print("------------Future Frames (Next 10 frames of the sequence)----------------")
        display(HBox(future))

        if pred_batch is not None:
            print("------------predicted frames(predicted 10 frames of the sequence)----------------")
            display(HBox(pred))