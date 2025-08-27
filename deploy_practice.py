import os, sys
import torch, numpy as np
import onnx
import onnxscript
import onnxruntime

print(f"onnxscript vesion: {onnxscript.__version__}\nonnxruntime version: {onnxruntime.__version__}\n\n")

# quantization
# import to .onnx, export to any other format (TF, Caffe etcc)

# onnx uses protobuf (protocol buffers)
# but onnx doesn't support every single NN layer...sometimes you'll have to write some layers yourself (in c/c++)

# torch.onnx() # captures torch computational graph, converts it into an onnx one
# which is faster?
    # torch model optimized with TorchScript and TensorRT
    # torch model -> onnx -> optimize with TensorRT 

# 1. Triton - a Python-based language for writing high-performance GPU kernels by OpenAI
## https://dhnanjay.medium.com/triton-gpu-programming-for-neural-networks-16271d729f78

# 2. Triton Inference Server - NVIDIA open-source project for deploying and serving AI models at scale in production 
# to communicate with AI applications, devs use HTTP/gRPC ednpoints

from train_model import Conv_DenoisingAutoenc, Linear_DenoisingAutoenc, \
                        add_layers_encoder, add_layers_decoder

import cv2 
im = 'historical_effect_method1.png'
# to gpu ?
device = torch.device('cuda')
args = (torch.tensor(cv2.resize(cv2.imread(im), (128,128), interpolation=cv2.INTER_AREA), dtype=torch.float32).to(device),)
print(args[0].shape)

# for a torch.Tensor, 'requires_grad' is False by default

model_path = 'tr_model.pt'
model = torch.load(model_path)
model.eval()
#output = model(torch.randn(128, 128, 3))
#print(f"output: {output}\n")

# needd args because ONNX export requires a sample input to trace the computation graph
# torch models can have dynamic behavior (conditionals, variable shapes, etc.)
    # so ONNX can’t just look at the code — it has to run the model once with example inputs to see what operations are used and how tensors flow
# without args (input example), ONNX doesn’t know what graph to export
args = torch.randn(128, 128, 3).to(device)

# USING torch.onnx.dynamo_export()
# onnx_model = torch.onnx.dynamo_export(model, args).save('denoising_convAutoencoder.onnx')
# getting an error because of BatchNorm (if there're batchnorm layers in the NN): https://github.com/pytorch/pytorch/issues/99662#issuecomment-1532224679

import torch.nn as nn
class ONNXCompatibleModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.downsampled_layers = original_model.downsampled_layers
        self.upsampled_layers = original_model.upsampled_layers
    
    def forward(self, input_image: torch.Tensor):
        input_image_prep = input_image.permute(-1, 0, 1).unsqueeze(0)  # (1, c, h, w)
        res = self.downsampled_layers(input_image_prep)
        res = self.upsampled_layers(res)
        return res

onnx_model = ONNXCompatibleModel(model)
onnx_model.eval()

torch.onnx.export(
    onnx_model,
    args,
    'denoising_convAutoencoder.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

# to open an ONNX graph, use Netron
# instead of using torch.onnx.export (old), use torch.onnx.dynamo_export() (new)

# aot autograd - speed up training of DL models 
    # easy integration with compilers cuz fwd and bwd passes graphs get pre-compiled - just like prog.languages' compilers 
# computation graphs are built dynamically in torch (by default) and statically in TF
# aot autograd pre-compiles the graphs (both forward and backward) before training starts - knows the entire computation flow upfront 
# + tensorboard


for i, par in enumerate(model.parameters()):
    print(f'par {i+1}: {par}')

a = torch.randn((2,3))
print(a.sum(), '\n', a.cos(), '\n', a.sin(), '\n', a.abs())


print('unbinding: ', torch.unbind(torch.tensor([[1,2,3],[3,4,5], [0,19,4]]), dim=-1), '\n')
big_tensor = torch.tensor([1,2,3])
sm1, sm2, sm3 = big_tensor.unbind() # 1 2 3 
# .detach() - detaching tensor from the current graph. The result will never require gradient

# logits = probability for all classes [0.2, 0.7, 0.1] cat dog car
# pickle vs .pt 

# use nn.Parameter() if torch.Tensor is trainable
# use self.register_buffer(name, tensor) if torch.Tensor isn't trainable

# torch.save() or pickle.dump() ?
# just use torch.save() cuz it's pickle under the hood + handles torch stuff properly

# autograd 
a = torch.tensor(3, dtype=torch.float16, requires_grad=True) # PyTorch tracks ops to build a graph
# requires_grad=True - compute gradients for this tensor
b = a-2 # wrt to a
print(f"b wrt a: {b.grad_fn(a)}") # db/da  (3?)
c = b*3*torch.tensor(1.5, dtype=torch.float16)
print(c) # (3-2)*3*1.5 = 4.5
print(b.abs())
print(f"\nc.grad_fn(b): {c.grad_fn(b)}\n\n")
c.backward() # computes the gradient of current tensor wrt graph leave
# val.backward() 'val' should always be a scalar 
print(f"c.grad_fn: {c.grad_fn}") # basically .grad_fn attribute is not used (we dont need to do anything with it)
print(f"gradient of a (dc/da) a.grad: {a.grad}") #


print('========='*9)
import torch
g = torch.tensor([1.0, 2.0], requires_grad=True)
h = g * 2
i = h + 3
p = i.sum()
# Call retain_grad() on 'b' to keep its gradient
h.retain_grad()
i.retain_grad()
p.backward()
print(f"g.grad: {g.grad}\n")  # Gradient for leaf tensor 'a' is available
print(f"h.grad: {h.grad}\n")  # Gradient for non-leaf tensor 'b' is now available
print(f"i.grad: {i.grad}\n")  # Gradient for non-leaf tensor 'c' is NOT available by default


print('========='*9)
x = torch.tensor(2., dtype=torch.float16, requires_grad=True)
a = torch.tensor([[4,2,5], [0,3,8]], dtype=torch.float16, requires_grad=True)
y = x**5
print(y.grad_fn(x)) 
# grad_fn is not a function you call – it’s a record of how a tensor was created (the operation node in the computation graph)
print(y.grad_fn(a))


# forward_y = img * torch.sin(weights)
# loss = abs(target_y - forward_y)
# loss.backward()

    # for all parameters in the model, .grad is saved and based on these optimizer.step() steps
    # w1.grad == loss_func'(w1)
    # optimizer.step(), update all parameters
        # w1 -= w1*grad(w1)
        # w2 -= w2*grad(w2)
        # ...
        # b903 -= b903*grad(b903)

# f'(x)
# x = parameter (weights/bias etc)
# so f'(parameter) = abs(target_y - img*torch.sin(parameter))'

print(f"type(onnx_model): {type(onnx_model)}\nonnx_model: {onnx_model}")
print('====== torch-ONNX conversion completed successfully ======')

if __name__ == '__main__':
    print('\n')
#    print(torch.utils.collect_env.get_pretty_env_info())
    print('\n')