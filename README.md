# Unofficial implementation of Video Ladder Networks

## Overview

Implementation of video prediction pipeline as [paper](https://arxiv.org/pdf/1612.01756.pdf) as a part of course project for Lab Vision Systems (MA-INF 4308). Model is evaluated on MMNIST and KTH dataset pre-trained weights for same are available

## Project Structure

```
.
├── checkpoints         # saving trained models
├── configs             # configs used for different experiments
├── data                # data directory 
│   └── KTH
│   └── MMNIST          
├── model_eval          # scripts for evaluating trained models
├── models              # building blocks of the model 
├── metrics_data        # stored data of computed metrics
├── results             # some images/GIFs computed from trained model
├── tboard_logs         # logs from our experiments
├── utils               # contains trainer to train model and some extra functionalities
└── eval.py             
└── train.py
└── summary.ipynb
```

## Installation

Please refer to [INSTALL.md](readme/INSTALL.md) for installation instructions

## Usage

First download pretrained weights for model as mentioned in above section.

Refer to this [notebook](summary.ipynb) to know about detailed usage of this repository.

## Results


**NOTE** Images padded with green are context frames, red padding indicated images predicted by our model

### Qualitative Results

MMNIST

![mmnist_1](images/all_frames1_random_batch_mmnist.gif)
![mmnist_2](images/all_frames2_random_batch_mmnist.gif)
![mmnist_3](images/all_frames3_random_batch_mmnist.gif)
![mmnist_4](images/all_frames4_random_batch_mmnist.gif)
![mmnist_5](images/all_frames5_random_batch_mmnist.gif)
![mmnist_6](images/all_frames6_random_batch_mmnist.gif)
![mmnist_7](images/all_frames7_random_batch_mmnist.gif)
![mmnist_8](images/all_frames8_random_batch_mmnist.gif)
![mmnist_9](images/all_frames9_random_batch_mmnist.gif)
![mmnist_10](images/all_frames10_random_batch_mmnist.gif)


KTH

![kth_1](images/all_frames1_random_batch_kth.gif)
![kth_2](images/all_frames2_random_batch_kth.gif)
![kth_3](images/all_frames3_random_batch_kth.gif)
![kth_4](images/all_frames4_random_batch_kth.gif)
![kth_5](images/all_frames5_random_batch_kth.gif)
![kth_6](images/all_frames6_random_batch_kth.gif)
![kth_7](images/all_frames7_random_batch_kth.gif)
![kth_8](images/all_frames8_random_batch_kth.gif)
![kth_9](images/all_frames9_random_batch_kth.gif)
![kth_10](images/all_frames10_random_batch_kth.gif)

### Quantitative Results

MMNIST

![quantitative_mmnist](images/plot_mmnist_compare.png)

KTH

![quantitative_kth](images/plot_kth_compare.png)


## Contact

Amit Rana (amit.rana@rwth-aachen.de), Dhagash Desai (dhagash.desai@uni-bonn.de), Lina Hashem (lina.gamal.hashem@gmail.com)


## Credits

We would like to thank [Angel Villar-Corrales](https://github.com/angelvillar96) for his guidance throughout the project.