# BASIC AUTOENCODER

# data - image/video/text(sometimes)/audio/

# learn compressed (encoded) representation of data and reconstruct the original data from that compressed version
# use-case: some random noisy image or video. 
    # train set: many 'clean'-original images of smth. And we want to get the original (reconstructed) image from a new, unclean image

## Encoder
# (multiple linear transformations) image * weights + bias 
# relu(multiple linear transformations)

## Hidden Layer (bottleneck - 'bridge' between encoder and decoder)
# it can be either undercomplete (fewer dimensions but more compact) or overcomplete (larger than the input but is capable of capturing more features)

## Decoder
# tries to reconstruct original data from hidden_layer_output 
# (multiple linear transformations) hidden_layer_output
# output layer - compare activationfunction((multiple linear transformations) hidden_layer_output) with original data

# ~/.cache/kagglehub/datasets/AI_for_Art_Restoration_2

# reshape images to one size = train_set
# add noises to each image: train_set_noise = [image1_gaussnoise, image1_randomnoise, image1_blur, image2_gaussnoise, ...]
# train an autoencoder: MSE(train_set_noise, train_set)

import cv2
import os, sys
import base64
import random as r 
import torch.nn as nn
from typing import Literal
import argparse
import tqdm

train_orig_path, train_noise_path = r"autoencoder_data/train_orig_2", r"autoencoder_data/train_noise"
print(len(os.listdir(train_orig_path)), len(os.listdir(train_noise_path)))

orig, noise = cv2.imread(os.path.join(train_orig_path, os.listdir(train_orig_path)[560])), \
            cv2.imread(os.path.join(train_noise_path, os.listdir(train_noise_path)[560]))
cv2.imshow('orig_561', orig); cv2.waitKey(0)
cv2.imshow('noise_561', noise); cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.IMWRITE_JPEG_QUALITY

## what is base64 code?

# learn codings - ascii, base64, uint8 (dtype) etc, for images
# jpeg - сжатый формат 

# compressing an image into an .jpeg image - nd.array
# jpeg can be compressed
res, encoded_im = cv2.imencode('.jpeg', orig, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
print(f"{res}\n{encoded_im.shape}{encoded_im.dtype}")
print(type(encoded_im))
base64_encoded = base64.b64encode(encoded_im)
print(type(base64_encoded), base64_encoded[-20])
ascii_decoded = base64_encoded.decode('ascii')
print(type(ascii_decoded), len(ascii_decoded))

image_shape = (800, 800, 3)
noise_ims = [cv2.imread(relative_path) for relative_path in [os.path.join(train_noise_path, im_p) for im_p in os.listdir(train_noise_path)[:20]]]
orig_ims = [cv2.imread(relative_path) for relative_path in [os.path.join(train_orig_path, im_p) for im_p in os.listdir(train_orig_path)[:20]]]
print("\norig images' shapes: ", [orig_im.shape for orig_im in orig_ims])
print("\nnoise images' shapes: ", [noise_im.shape for noise_im in noise_ims])


#sys.exit(0)
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim
print(torch.__version__, torch.cuda.is_available())

orig_images_p_tr, noise_images_p_tr = r"autoencoder_data/train_orig_2", r"autoencoder_data/train_noise"
orig_images_p_te, noise_images_p_te = r"autoencoder_data/test_orig", r"autoencoder_data/test_noise"


class AutoEncoder(nn.Module):
    def __init__(self, image_dims, compress_size):
        super(AutoEncoder, self).__init__()
        self.image_dims, self.compress_size = image_dims, compress_size

        self.dropout = nn.Dropout()
        self.leaky_relu = nn.LeakyReLU()
        self.encoder = nn.Sequential([
            F.linear(self.image_dims, 200),
            self.leaky_relu(),
            self.dropout(),
            F.linear(200, 100),
            self.leaky_relu(),
            F.linear(100, self.compress_size),
            self.leaky_relu()
        ])
        
        # latent state representation
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential([
            F.linear(self.compress_size, 50),
            self.sigmoid(),
            F.linear(50, 100),
            F.linear(100, 150),
            self.sigmoid(),
            self.dropout(),
            F.linear(150, 200),
            F.linear(200, self.output_size),
            self.relu()
        ])


    def forward(self, input_image):
        assert type(input_image)==torch.Tensor, 'input image has to be a torch.Tensor'
        assert torch.tensor(input_image.shape).tolist()==list(image_shape), 'input image should be of shape (800, 800, 4)'
        print(f"doing forward on image of shape {input_image.shape}")
        return self.decoder(self.encoder(input_image))


# do autoencoders use CNNs? i have 3d images so do either: flattern*weights->unflatten or cnn(3d)
# we use a CAE - convolutional autoencoder - it's specifically designed for image data
# in a CAE, encoder layers are called conv.layers while decoder layers - deconv.layers


# 2 modes (architectures): 'linear' (flatten * weights -> unflatten)
                        #  'conv' (conv layers(downsampling) -> ... -> transposed conv layers(upsampling))
# transposed kernel -> the opposite of downsampling: array*kernel -> original_array (bigger)


# only works if ther're images in 'train_noise' and 'train_orig' folders
# images in train -> choose random X [0.1; 0.9] -> add them to test folders
class Dataset:
    '''
    Class to form an image dataset and create train/test splits:

    - get images from only train folders
    - split image list into train/test (randomly)
    - add images*X є [0.1; 0.9] to the test folders
    - it will return tuple[list[torch.Tensor]] (train_noise, train_orig, test_noise, test_orig)
    '''
    # all pars are str
    def to_torchTensor(self, ntr, otr, nte, ote) -> tuple[list]:
        def load_images(base_path, file_list):
            return [torch.tensor(cv2.imread(os.path.join(base_path, im), cv2.IMREAD_UNCHANGED), dtype=torch.float32) for im in file_list]

        if type(ntr)==str and type(otr)==str and type(nte)==str and type(ote)==str:  # folder paths
            ntr_l = load_images(ntr, os.listdir(ntr))
            otr_l = load_images(otr, os.listdir(otr))
            nte_l = load_images(nte, os.listdir(nte))
            ote_l = load_images(ote, os.listdir(ote))
        else: 
            raise ValueError("to_torchTensor() must receive folder paths")
        return ntr_l, otr_l, nte_l, ote_l


    # all pars are strs (image paths)
    def train_test_split(self, noise_train, orig_train, noise_test, orig_test, train_split_ratio: float) -> tuple[list[torch.Tensor]]:
        assert 0.1 <= train_split_ratio <= 0.9, "'train_split_ratio' should be in range [0.1, 0.9]"
        # split images, cv2.imwrite() to test paths
        n_train_files = r.choices(os.listdir(noise_train), k=int(len(os.listdir(noise_train)) * train_split_ratio))
        o_train_files = [otrain_im for otrain_im in os.listdir(orig_train) if otrain_im in n_train_files]
        n_test_files = [noise_im for noise_im in os.listdir(noise_train) if noise_im not in n_train_files]
        o_test_files = [orig_im for orig_im in os.listdir(orig_train) if orig_im not in o_train_files]
        print(f"len(n_train) {len(n_train_files)}\n len(o_train): {len(o_train_files)}\n len(n_test): {len(n_test_files)}\n len(o_test): {len(o_test_files)}\n")
        try:
            for im in n_test_files:
                cv2.imwrite(os.path.join(noise_test, im), cv2.imread(os.path.join(noise_train, im), cv2.IMREAD_UNCHANGED))
            for im in o_test_files:
                cv2.imwrite(os.path.join(orig_test, im), cv2.imread(os.path.join(orig_train, im), cv2.IMREAD_UNCHANGED))
            print(f"\ntest images have been added to folders {noise_test} and {orig_test}\n")
            # pass folder paths 
            n_train, o_train, n_test, o_test = self.to_torchTensor(noise_train, orig_train, noise_test, orig_test)
            print('\nsuccess...returning torch.Tensors: \n')
            return n_train, o_train, n_test, o_test
        except Exception as e:
            print('error: ', e)
            return [], [], [], []  


    def __init__(self, 
                train_orig_ims_p, train_noise_ims_p, 
                test_orig_ims_p, test_noise_ims_p, train_split: float):
        
        self.train_noise_ims, self.train_orig_ims, \
        self.test_noise_ims, self.test_orig_ims = self.train_test_split(train_noise_ims_p, train_orig_ims_p,
                                                                            test_noise_ims_p, test_orig_ims_p, train_split)
        
    def __iter__(self):
        return iter((self.train_noise_ims, self.train_orig_ims, self.test_noise_ims, self.test_orig_ims))

    # dont need __len__ because __iter__ already returns lists
    # def __len__(self, data): # list[torch.Tensor]
    #     return len(data)


device = torch.device('cuda')
train_noisy_ims, train_clean_ims, test_noisy_ims, test_clean_ims = Dataset(orig_images_p_tr, noise_images_p_tr, orig_images_p_te, noise_images_p_te, 0.75)
train_noisy_ims, train_clean_ims, test_noisy_ims, test_clean_ims = train_noisy_ims[::3], train_clean_ims[::3], test_noisy_ims[::3], test_clean_ims[::3]
train_noisy_ims, train_clean_ims, test_noisy_ims, test_clean_ims = train_noisy_ims.to(device), train_clean_ims.to(device), \
                                                                   test_noisy_ims.to(device), test_clean_ims.to(device)

__all__ = [train_noisy_ims, train_clean_ims, test_noisy_ims, test_clean_ims]
sys.exit(0)
# print(len(train_noisy_ims)) # list[Tensor]
# print(train_noisy_ims[9])

# the most common loss function for autoencoders is MSE