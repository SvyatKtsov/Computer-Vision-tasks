import os, sys
import torch, torch.nn as nn
import torch.nn.init as nn_init  # ?
import numpy as np
import cv2, ffmpeg, matplotlib.pyplot as plt
import argparse
import urllib # HTTP Python client


# import gstreamer 
device = torch.device('cuda:0')
from train_model import Conv_DenoisingAutoenc, Linear_DenoisingAutoenc, \
                        add_layers_encoder, add_layers_decoder


def eval(model_path, test_noiseImages_path, images_count: int = 6) -> None:
    # images_count - number of images to test (from test_noiseImages_path)
    try:
        model = torch.load(model_path).to(device) # we can also load only the weights
        model.eval()
        # for model_par in Conv_DenoisingAutoenc().state_dict():
        #     print(f"model param tensor: {model[model_par]}\n \
        #         its shape: {model[model_par].shape}\n\n")
        # do the same but for optimizer - print(optimizer.state_dict()[optim_param].shape)
        
        for i in range(images_count):
            im = cv2.imread(os.path.join(test_noiseImages_path, os.listdir(test_noiseImages_path)[i]), cv2.IMREAD_UNCHANGED) 
            if i==0: print(f"im.shape: {im.shape}") # should be torch.Size([1, 3, H, W])
            #im = im.squeeze(0).permute(1, -1, 0).numpy(force=True).astype(np.uint8)
            #im = torch.tensor(im, dtype=torch.float32) / 255.0  # normalize if model trained that way
            output_image = model(torch.tensor(im, dtype=torch.float32).to(device))
            print(f"\nwhether output_image requires_grad: {output_image.requires_grad}\n")
            if output_image.requires_grad==True: 
                output_image = output_image.numpy(force=True)
                output_image = np.transpose(output_image.squeeze(0), (1,2,0))
                print(f"output_image.shape after squeeze and permute: {output_image.shape}")
                if i==0: print(f"max min: {output_image.min(), output_image.max()}")
            # if i==0: 
            #     print(f"output_image[0].shape (model(input_image)): {output_image.size()}")
            #     print(f"output min, output max: {output_image.min()}, {output_image.max()}\n")

            #output_image_disp = output_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#            if i==0: print(f"output_image[0].shape output_image.squeeze(0).permute(1, 2, 0): {output_image_disp.shape}")
            plt.subplot(2, images_count, i+1)
            orig_im = cv2.cvtColor(cv2.imread(os.path.join('autoencoder_data/test_orig', os.listdir('autoencoder_data/test_orig')[i])),
                                cv2.COLOR_BGR2RGB)
            plt.imshow(orig_im); plt.title(f'original image {i+1}')
            plt.subplot(2, images_count, i+images_count+1)
            #plt.imshow(cv2.cvtColor((output_image_disp*255).astype(np.uint8), cv2.COLOR_BGR2RGB)); plt.title(f'noise image {i+1}')
            #plt.imshow(output_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)); plt.title(f'noise image {i+1}')
            plt.imshow(output_image.astype(np.uint8)); plt.title(f'noise image {i+1}')
            

        plt.show()

    except Exception as e:
        print(f"exception when loading {model_path}:\n{e}")
        

# TF is faster for inference + there're frameworks for edge devices

# QUANTIZATION, torch lightning
# YOLOv8 quantization https://medium.com/@sulavstha007/quantizing-yolo-v8-models-34c39a2c10e2
# inference == post-training

# do not add a softmax prediction layer, this is included in the cross entropy loss function or it'll lead to bad results

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('model_path', type=str)
    cli.add_argument('noise_ims_path', type=str)
    cli.add_argument('images_count', type=int)
    args = cli.parse_args()
    print('\n======== INFERENCE STARTED ========\n')
    eval(args.model_path, args.noise_ims_path, args.images_count)
    print('\n======== INFERENCE FINISHED ========\n')