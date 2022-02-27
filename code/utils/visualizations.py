import numpy as np
import io
import imageio
from IPython.display import display
from ipywidgets import widgets, HBox, VBox, Layout, Box

def show_gifs(sample_batch, nsamples = 5):
    '''
    sample_batch: batch of sequences shape:(seq_len, batch_size, nc, h, w)
    nsamples: #sequences to display
    '''
    # Reverse process before displaying
    sample_batch = sample_batch.cpu().numpy() * 255.0    
    sample_batch = sample_batch.squeeze(2)
    
    # box_layout = Layout(flex_flow='row')
    for i in range(nsamples):
        video = sample_batch[:,i,:]
        imageio.mimsave(f"past/past_frames{i+1}.gif",video[:10].astype(np.uint8),"GIF",fps=5)
        imageio.mimsave(f"future/future_frames{i+1}.gif",video[10:].astype(np.uint8),"GIF",fps=5)
        
    past = []
    future = []
    for i in range(1,6):
        past.append(widgets.Image(value=open(f"past/past_frames{i}.gif", 'rb').read()))
        future.append(widgets.Image(value=open(f"future/future_frames{i}.gif", 'rb').read()))
    # imageA = widgets.Image(value=open('real1.gif', 'rb').read())
    # imageB = widgets.Image(value=open('real2.gif', 'rb').read())

    # hbox = HBox([imageA, imageB])
    # display(hbox)
    print("------------Past Frames (First 10 frames of the sequence)----------------")
    display(HBox(past))

    print("------------Future Frames (Next 10 frames of the sequence)----------------")
    display(HBox(future))