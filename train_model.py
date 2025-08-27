import torch, numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import argparse
import os, sys
from typing import Literal
from tqdm import tqdm
import cv2

device = torch.device('cuda')
def images_to_tensor(base_path, file_list):
    return [torch.tensor(cv2.imread(os.path.join(base_path, im), cv2.IMREAD_UNCHANGED), dtype=torch.float32).to(device)
             for im in file_list]

train_noisy_ims = images_to_tensor('autoencoder_data/train_noise', sorted(os.listdir('autoencoder_data/train_noise')))
train_clean_ims = images_to_tensor('autoencoder_data/train_orig_2', sorted(os.listdir('autoencoder_data/train_orig_2')))
test_noisy_ims = images_to_tensor('autoencoder_data/test_noise', sorted(os.listdir('autoencoder_data/test_noise')))
test_clean_ims = images_to_tensor('autoencoder_data/test_orig', sorted(os.listdir('autoencoder_data/test_orig')))

modes = Literal['conv', 'linear']

class Conv_DenoisingAutoenc(nn.Module):
    def __init__(self, encoder_layers, dec_neurons, kernel, last_encoder_outchannels, in_channels, num_down):
        super(Conv_DenoisingAutoenc, self).__init__()
        # conv2d relu maxpool
        # encLayersNum conv2d; 
        # relu should be used after each conv layer (while extracting features)
        # p-value - ...
        # for upsampling, we can use either interpolation (bilinear/cubic) or deconvolution 
        # (try both using --upsampling_type 'deconv' and 'bilinear'/'bicubic') 
        # deconv is LEARNABLE - unlike interpolation 
        #!!!!!!!!!!!!!!!!!! Wiener deconvolution
        self.downsampled_layers = nn.Sequential(*encoder_layers)
        last_conv = next(layer for layer in reversed(encoder_layers) if isinstance(layer, nn.Conv2d))
        last_encoder_outchannels = last_conv.out_channels
        self.upsampled_layers = nn.Sequential(
            *add_layers_decoder(dec_neurons, kernel, last_encoder_outchannels, in_channels, num_down)
        )

        # literally the opposite of conv - nn.ConvTranspose2d(64, 32), nn.ConvTranspose2d(32, 3) 
        # get encoder layers and reverse it (?) # answer - no you don't have to mirror the encoder but it's a 'convention' to write like this (?) 
           ## BUT Be careful, if your decoder is too powerful and you have a VAE the decoder might choose to ignore the latent codes entirely 
           ## so a more complex decoder may be an overkill

        # self.sigmoid = nn.Sigmoid() # ?
        # usually no activation function after decoder(encoder(input_image))
        # nn.Identity() ? example:
        #  It can be used as a placeholder in case you want to change smth
            # batch_norm = nn.BatchNorm2d
            # if dont_use_batch_norm:
            #     batch_norm = nn.Identity
        # nn.InstsanceNorm2d(..., affine=True) or nn.BatchNorm ?

        # @lru_cache
        # assert cond, mess; mess is optional 

    def forward(self, input_image: torch.Tensor):
        # h, w, c
        
        assert torch.is_tensor(input_image) and input_image.ndim == 3 and input_image.shape[-1] in (3, 4), \
            'input_image should be a 3d-tensor and have 3 or 4 (first) channels - (h, w, c)'
        
        input_image_prep = input_image.permute(-1, 0, 1).unsqueeze(0) # (1, c, h, w)
        res = self.downsampled_layers(input_image_prep)
        res = self.upsampled_layers(res)
        assert res.shape == input_image_prep.shape, \
            f"decoder_output.shape {res.shape} != input_image shape {input_image_prep.shape}, but should be =="
        return res


class Linear_DenoisingAutoenc(nn.Module): 
    def __init__(self, m):
        super(Linear_DenoisingAutoenc, self).__init__()
        pass


def add_layers_encoder(neurons_each_layer, kernel, in_channels=3):
    layers = []
    downsamples = 0
    current_channels = in_channels
    
    for i, out_channels in enumerate(neurons_each_layer):
        layers.append(nn.Conv2d(current_channels, out_channels, kernel, padding=kernel//2))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=False))
        current_channels = out_channels  
        
        # maxpool after conv layer (except for the last one)
        if i < len(neurons_each_layer) - 1:  
            layers.append(nn.MaxPool2d(2))
            downsamples += 1
    
    return layers, downsamples


def add_layers_decoder(neurons_each_layer, kernel, latent_channels, out_channels, num_upsamples):
    layers = []
    prev_channels = latent_channels
    ups_left = num_upsamples

    for ch in neurons_each_layer:
        # stride=2 for upsampling when we have upsamples left
        stride = 2 if ups_left > 0 else 1
        layers.append(nn.ConvTranspose2d(prev_channels, ch, kernel, stride=stride,
                                         padding=kernel//2, output_padding=(1 if stride==2 else 0)))
        layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.ReLU(inplace=False))
        prev_channels = ch
        if stride == 2:
            ups_left -= 1

    # last layer to reconstruct the original number of channels
    stride = 2 if ups_left > 0 else 1
    layers.append(nn.ConvTranspose2d(prev_channels, out_channels, kernel, stride=stride,
                                     padding=kernel//2, output_padding=(1 if stride==2 else 0)))
    return layers


def train(model_architecture: str, encLayersNum, enc_neurons: list, decLayersNum, dec_neurons: list, 
          kernel, epochs, save_filename=''):
    # if 'model_architecture' is not 'conv' nor 'linear', it gives a type error
    # + split train set into train and val sets (for training monitoring and hyperparameter optimization(? - check if this is true)) 
    
    # how are images converted from analogue to digital ? 
    # process that involves sampling, quantization, and encoding. This process is handled by an analog-to-digital converter (ADC)
    # for CV tasks, probably BatchNorm/InstanceNorm is better (?)
    ## BatchNorm is used to make gradient descent convergence faster
    enc_neurons = list(map(int, args.neurons_Enc.split(',')))
    dec_neurons = list(map(int, args.neurons_Dec.split(',')))
    assert encLayersNum == len(enc_neurons) and decLayersNum == len(dec_neurons), 'mismatch between layer num and neurons num for enc or dec'

    batch_size = 8
    device = torch.device('cuda')

    # figure out input channels from data (3 or 4)
    in_channels = train_noisy_ims[0].shape[-1]

    if model_architecture == 'conv':
        encoder_layers, num_down = add_layers_encoder(enc_neurons, kernel, in_channels=in_channels)
        conv_model = Conv_DenoisingAutoenc(encoder_layers, dec_neurons, kernel, 
                                           encoder_layers[-1].out_channels if isinstance(encoder_layers[-1], nn.Conv2d) else None,
                                           in_channels, num_down).to(device)
        optimizer = optim.Adam(conv_model.parameters(), lr=0.0015)  
        loss_f = nn.MSELoss()
        
        epoch_losses = []
        conv_model.train()
        
        for epoch in range(epochs):
            epoch_total_loss = 0
            num_batches = (len(train_noisy_ims) + batch_size - 1) // batch_size  # ceiling division
            
            with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx in range(0, len(train_noisy_ims), batch_size):
                    batch_loss = 0
                    optimizer.zero_grad()
                    
                    batch_noisy = train_noisy_ims[batch_idx:batch_idx+batch_size]
                    batch_clean = train_clean_ims[batch_idx:batch_idx+batch_size]
                    
                    for i, (image_noise, image_orig) in enumerate(zip(batch_noisy, batch_clean)):
                        image_noise, image_orig = image_noise.to(device), image_orig.to(device)
                        forward_output = conv_model(image_noise)
                        if epoch==epochs-1 and i<16: print(f"epoch 0, min max: {torch.min(forward_output).detach().item(), \
                                                                         torch.max(forward_output).detach().item()}")
                                                # print 16 images' min and max values
                        # print the last forward pass image
                        if epoch==epochs-1 and i==batch_size-1: # [image_0,...image_batchsize-1] batch_size==8
                            #print(f"last forward_output.shape: {forward_output.shape}\n"); sys.exit(0)

                            # cv2.imshow('last forward pass img', 
                            #         forward_output.squeeze(0).permute(1, -1, 0).numpy(force=True).astype(np.uint8)) 

                            # force=True - same as 't.detach().cpu().resolve_conj().resolve_neg().numpy()'
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        loss = loss_f(forward_output, image_orig.permute(-1,0,1).unsqueeze(0))  # match (1,c,h,w)
                        batch_loss += loss
                    
                    batch_loss.backward()
                    optimizer.step()
                    epoch_total_loss += batch_loss.item()
                    pbar.update(1)
            
            avg_epoch_loss = epoch_total_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            print(f"loss: {avg_epoch_loss:.6f}\n")
            if epoch==epochs-1:
                plt.plot(epoch_losses, [*range(epochs)])
                plt.title(f'Convolutional Autoencoder train loss\nepochs: {epochs}\nbatches: {batch_size}\n \
                            enc layers num: {encLayersNum}\ndec layers num: {decLayersNum}')
                plt.xlabel('Epochs'); plt.ylabel('Epoch loss')
                plt.show()
    

    else:
        linear_model = Linear_DenoisingAutoenc(None).to(device)
    
    if save_filename:
        try:
            torch.save(conv_model if model_architecture == 'conv' else linear_model, save_filename) 
            print(f"Model saved to {save_filename}")
        except Exception as e:
            print('exception %s' % e, '\n')




__all__ = [Conv_DenoisingAutoenc, Linear_DenoisingAutoenc, add_layers_encoder, add_layers_decoder]

def test_model_speed(model_func_obj, noise_im: list[torch.Tensor], orig_im: list[torch.Tensor], mode):
    assert mode in ['eager', 'aot'], "mode should be either 'eager' or 'aot'"
    assert len(noise_im)==1, 'testing eager/aot speed only on one tensor(image)'
    
    import time
    device = noise_im[0].device
    is_cuda = device.type == 'cuda'

    warmup = 10
    iters = 100 
    loss_f = nn.MSELoss()

    print(f"--- Warming up {mode} mode ---")
    for _ in range(warmup):
        res_forward = model_func_obj(noise_im[0])
        # output from a Conv model is usually (B, C, H, W). 
        # orig_im[0].permute(-1,0,1).unsqueeze(0) -> (1, C, H, W)
        target = orig_im[0].permute(2, 0, 1).unsqueeze(0)
        loss = loss_f(res_forward, target)
        loss.backward()
        # clear grads for the next warmup iteration
        for p in m.parameters():
            if p.grad is not None:
                p.grad.zero_()
        if noise_im[0].grad is not None:
            noise_im[0].grad.zero_()


    fw_latencies, bw_latencies = [], []
    print(f"--- Benchmarking {mode} mode ({iters} iterations) ---")
    
    for _ in range(iters):
        for p in m.parameters():
             if p.grad is not None:
                p.grad.zero_()
        if noise_im[0].grad is not None:
            noise_im[0].grad.zero_()

        if is_cuda:
            torch.cuda.synchronize()
        fw_begin = time.perf_counter()

        res_f = model_func_obj(noise_im[0])
        
        if is_cuda:
            torch.cuda.synchronize()
        fw_end = time.perf_counter()

        loss = loss_f(res_f, target)
        
        if is_cuda:
            torch.cuda.synchronize()
        bw_begin = time.perf_counter()
        
        loss.backward()
        
        if is_cuda:
            torch.cuda.synchronize()
        bw_end = time.perf_counter()

        # seconds
        fw_latencies.append((fw_end - fw_begin) * 100)
        bw_latencies.append((bw_end - bw_begin) * 100)
    
    avg_fw, avg_bw  = sum(fw_latencies) / len(fw_latencies), sum(bw_latencies) / len(bw_latencies)

    print(f'\nResults for [{mode}]:'
          f'\n\tAvg Forward: {avg_fw:.3f} ms'
          f'\n\tAvg Backward: {avg_bw:.3f} ms\n')


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('archit', type=str)
    cli.add_argument('encLayers_num', type=int)
    cli.add_argument('neurons_Enc', type=str, help="Comma-separated encoder neurons")
    cli.add_argument('decLayers_num', type=int)
    cli.add_argument('neurons_Dec', type=str, help="Comma-separated decoder neurons")
    cli.add_argument('kernel', type=int)  
    cli.add_argument('epochs', type=int)
    cli.add_argument('--save_filename', type=str)
    args = cli.parse_args()    
    print('\n============ TRAINING STARTED ============\n')
    train(args.archit, args.encLayers_num, args.neurons_Enc, 
          args.decLayers_num, args.neurons_Dec, args.kernel,
          args.epochs, args.save_filename)
    print('\n============ TRAINING FINISHED ============\n\n')
    print('comparing Eager (dynamic graph creation) with AOT (pre-compiled forward+backward graphs):\n')

    def model_func(model_classObj):
        def fn(x: torch.Tensor):
            #print(f"input_data.shape: {x.shape}, model_classObj: {model_classObj}\n")
            return model_classObj(x)
        return fn


    enc_neurons = list(map(int, args.neurons_Enc.split(',')))
    dec_neurons = list(map(int, args.neurons_Dec.split(',')))
    in_channels = train_noisy_ims[0].shape[-1]
    encoder_layers, num_down = add_layers_encoder(enc_neurons, args.kernel, in_channels=in_channels)
    m = Conv_DenoisingAutoenc(
        encoder_layers, dec_neurons, args.kernel, 
        encoder_layers[-1].out_channels if isinstance(encoder_layers[-1], nn.Conv2d) else None,
        in_channels, num_down).to(device)
    
    for noisy_im in train_noisy_ims[:1]:
        noisy_im.requires_grad = True

    test_model_speed(model_func(m), train_noisy_ims[:1], train_clean_ims[:1], 'eager')


    from functorch.compile import ts_compile, aot_function
    from functorch import make_functional_with_buffers
    
    # from torch._subclasses.fake_tensor import FakeTensorMode

    # aot_function needs to build a graph of your model before running it
    # it does this by tracing it with FakeTensors (placeholders with shape/dtype but no real data)

    params = dict(m.named_parameters())
    buffers = dict(m.named_buffers())

    fmodel, params, buffers = make_functional_with_buffers(m)

    # "stateless" function to be compiled - only pars and buffers
    def stateless_model_func(params, buffers, x):
        return fmodel(params, buffers, x)

    aot_nnc_fc = aot_function(
        stateless_model_func,
        fw_compiler=ts_compile,
        bw_compiler=ts_compile
    )

    # calls function with the *real* (not just shapes and dtypes) params and buffers
    def aot_model(x):
        return aot_nnc_fc(params, buffers, x)

    test_model_speed(aot_model, train_noisy_ims[:1], train_clean_ims[:1], 'aot')


