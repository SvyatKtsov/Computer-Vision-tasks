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



## PAINTING RECONSTRUCTION
# add cli args (so that i could train and test the model, for instance: python3 autoencoder.py 200 100 50 (neurons_num in the first second third layers of encoder/decoder))

# ~/.cache/kagglehub/datasets/AI_for_Art_Restoration_2

# check images, reshape to one size = train_set
# add noises to each image: train_set_noise = [image1_gaussnoise, image1_randomnoise, image1_blur, image2_gaussnoise, ...]
# train an autoencoder: MSE(train_set_noise, train_set)
# add CLI args + deploy somehwere, convert to ONNX, tensorboard etc, try optimization (quantization, pruning, runtime)

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import torch.optim as optim
print(torch.__version__, torch.cuda.is_available())

data_path = ...

class AutoEncoder(nn.Module):
    def __init__(self, image_dim, embedd_size):
        super(AutoEncoder, self).__init__()
        self.image_dim, self.embedd_size = image_dim, embedd_size

        self.dropout = nn.Dropout()
        self.leaky_relu = nn.LeakyReLU()
        self.encoder = nn.Sequential([
            F.linear(image_dim, 200),
            self.leaky_relu(),
            self.dropout(),
            F.linear(200, 100),
            self.leaky_relu(),
            F.linear(100, self.embedd_size),
            self.leaky_relu() # or self.leaky_relu ?
        ])
        
        # latent state representation
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Sequential([
            F.linear(self.embedd_size, 50),
            self.sigmoid(),
            F.linear(50, 100),
            F.linear(100, 150),
            self.sigmoid(),
            self.dropout(),
            F.linear(150, 200),
            F.linear(200, self.output_size),
            self.sigmoid()
        ])


    def forward(self, input_image):
        assert type(input_image)==torch.Tensor, 'input images have to be tensors'
        print(f"doing forward on image of shape {input_image.shape}")
        return self.decoder(self.encoder(input_image))


class Dataset:
    def __init__():
        pass

    def __len__():
        pass

    def __getitem__():
        pass


autoencoder = AutoEncoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.03)
loss_f = nn.MSELoss()
# output = loss_f(input, target)
# output.backward() 

autoencoder.train()

autoencoder.eval()