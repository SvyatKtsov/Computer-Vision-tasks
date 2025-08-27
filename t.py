import torch
import torch.nn as nn
import cv2, numpy as np
import matplotlib, matplotlib.pyplot as plt
from typing import Literal

a = torch.rand(2,3)
print(a, type(a), a.size(), a.shape, '\n', nn.Sigmoid()(a))

import os, sys
print(os.path.split('~Flispark/code/backend/file.txt'))
print(os.path.splitext('~Flispark/code/backend/file.txt'))
sys.exit(0)

# import kagglehub
# path = kagglehub.dataset_download("pes1ug22am047/damaged-and-undamaged-artworks")
# print("Path to dataset files:", path)


# import matplotlib.pyplot as plt
# categories = ['Apples', 'Bananas', 'Oranges', 'Grapes']
# sales = [150, 200, 120, 180]
# plt.bar(categories, sales)
# plt.title('Fruit Sales Data')
# plt.xlabel('Fruit Type')
# plt.ylabel('Number of Sales')
# plt.show()


# learn color theory, opencv basics...camera basics etc
# check .py files 'opencv_practice'


# waitKey(0) will display the window infinitely until any keypress (it is suitable for image display)
# waitKey(1) will display a frame for 1 ms, after which display will be automatically closed. Since the OS has a minimum time between switching threads, the function will not wait exactly 1 ms, it will wait at least 1 ms, depending on what else is running on your computer at that time
# So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1) the function will show a frame for at least 1 ms only

## OPENCV USES BGR WHILE OTHER CV LIBS USE RGB
p = r"autoencoder_data/train_set/1st-art-oil-painting-reproduction_vertical_and_hor_fl.jpg"
grayscaled_im = cv2.imread(p, 0) # default is cv2.IMREAD_COLOR or 1 (all images are opened as BGR in opencv by default)
# cv2.IMREAD_UNCHANGED  or -1 (BGRa, a-alpha...0-255 transparency for the image)
# cv2.IMREAD_GRAYSCALE  or 0
cv2.imshow('1', grayscaled_im); cv2.waitKey(0) 
print(f"ord('😅'): {ord('😅')}") # unicode - ASCII + other symbols (all languages/emojis...)
print(f"\ncv2.waitKey(0): {cv2.waitKey(0)}\n")
print(cv2.waitKey(0) & 0xFF == ord('q')) 
print(bin(7), bin(6))
print(7 & 6) # 2^1 + 2^2
# 6

# hex -> dec
print(int('0xFF', 16), True & True, 0b001 & 0b110, 0b001 and 0b110)
# 32-bit & 8-bit == 113; so 0xFF is like a mask - it gets only 8 bits of some number
print(bin(90))
print(ord('q'))

# if you need transparency (see-through parts), use PNG images cuz JPG/JPEG dont have it 
# '-1' has 1 additional channel, transparency
tr_png = cv2.imread('image.png', -1)
cv2.imshow('2', tr_png); cv2.waitKey(0)
cv2.destroyAllWindows()

original_im = cv2.imread(p, 1)
print(f"original_im type: {type(original_im)}")
unchanged_im = cv2.imread(p, -1)
im_l = [grayscaled_im, original_im, unchanged_im]
im_l.append(cv2.cvtColor(original_im, cv2.COLOR_BGR2HSV))  # COLOR_BGR2HSV COLOR_BGR2RGB
im_l.append(cv2.cvtColor(original_im, cv2.COLOR_BGR2HLS))
for i in range(len(im_l)):
    plt.subplot(1, len(im_l), i+1)
    plt.title(i+1)
    if i==0: 
        plt.imshow(im_l[i], cmap='gray')
        cv2.imwrite('grayscaled_img.jpg', im_l[i])
        cv2.imwrite('grayscaled_img.png', im_l[i])
    else: 
        plt.imshow(im_l[i])
plt.show()

## OR this:
# fig, axs = plt.subplots(1, len(im_l))
# for i, img in enumerate(im_l):
#     axs[i].imshow(img, cmap='gray' if i == 0 else None)
#     axs[i].set_title(str(i + 1))
# plt.show()

print(f"jpg vs png sizes: {os.path.getsize('grayscaled_img.jpg')/1000000}, {os.path.getsize('grayscaled_img.png')/1000000}")
# 1 kb = 1000 bytes
# 1 mb = 1000 kb


#### VIDEOS
vid_capture_object = cv2.VideoCapture(r"/home/skrand/Downloads/3571264-uhd_3840_2160_30fps.mp4")
# if not vid_capture_object.isOpened(): print('video couldnt be opened')
# else: 
#     fps = vid_capture_object.get(5)
#     print(f"fps: {cv2.CAP_PROP_FPS}, {fps}")

while (vid_capture_object.isOpened()):
    frame_returned, frame = vid_capture_object.read()
    h, w, c = frame.shape    
    #print(f"type(frame): {type(frame)}\n")  # np.ndarray
    if frame_returned:
        cv2.imshow(f'video_frame{h} {w} {c}', cv2.resize(frame, (1300, 700), cv2.INTER_LINEAR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()

gr = cv2.cvtColor(cv2.imread('grayscaled_img.jpg'), cv2.COLOR_BGR2RGB) # even if we open a gray-colored (1 channel) image, it's still gonna be 3 channels - read by OpenCV (BGR)
cv2.imshow('h', gr); cv2.waitKey(0)
print(f"gr.shape: {gr.shape}\n")
print(f"gr: {gr}\n")

def show_resized(title_prefix, img, fx, fy):
    methods = {
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_NEAREST': cv2.INTER_NEAREST, 
    }
    # to enlarge -> INTER_LINEAR or INTER_CUBIC (cubic is computationally harder (so slower) but the quality is higher)
    # to shrink -> INTER_AREA
    for name, method in methods.items():
        resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=method)
        cv2.imshow(f'{title_prefix} {name}', resized)


while True:
    show_resized('.resize() with scaling factors', gr, 1.3, 1.3)
    if (cv2.waitKey(0) & 0xFF == ord('q')): break
cv2.destroyAllWindows()


# CROPPING IMAGES
print('cropping images: \n')
print(f"gr.shape: {gr.shape}\n") # 800, 800, 3
cv2.imshow('cropped_gr', gr[350:450, 350:450]); cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.destroyWindow('cropped_gr')


M = 76 # h
N = 104 # w
x1 = 0
y1 = 0
gr_h, gr_w = gr.shape[:2]# for np arrays it's always H, W
# but for cv2.resize(), cv2.imshow() and other opencv methods/functions -> W,H
image_copy = gr.copy() # if one changes, the other doesn't change, unlike with .view()

# M x N
for y0 in range(0, gr_h, M):
    for x0 in range(0, gr_w, N):
        if (gr_h - y0) < M or (gr_w - x0) < N:
            break
        y1, x1 = min(y0 + M, gr_h), min(x0 + N, gr_w)
        tile = image_copy[y0:y1, x0:x1]
        cv2.imwrite(f'r/tile_{x0}_{y0}.jpg', tile)
        cv2.rectangle(image_copy, (x0, y0), (x1, y1), (0, 255, 0), 1)


cv2.imshow("Patched Image",image_copy)
cv2.imwrite("patched.jpg",image_copy)
cv2.waitKey(0)
cv2.destroyWindow('Patche Image')

# task: do smth similar
# дано не размер каждой ячейки, а их количество 
# и нужно написать алгоритм который нарисует все ячейки"автоматически", исходя из кол-ва ячеек
def divide_eq_parts(image, n, mode=Literal['eq', 'not_eq']) -> np.ndarray: # 1 if successful, 0 otherwise
    '''
    Returns the same image with n rectangles drawn in a specified mode
    '''
    assert type(image)==np.ndarray, "before dividing the image, make sure it's an np.ndarray"
    h, w = image.shape[:2]
    # if h or w isn't even (for ex w=863) then we won't get it divded equally BUT we can do either:
        # - 'eq' mode: do not 863 but 862 (if w%2!=0: w=-1)
        # - 'not_eq' mode: OR do all rectangles equal and then the last ones slightly bigger/smaller
    pass
    

# ROTATION, TRANSLATION
cv2.imshow('gr_2', gr); cv2.waitKey(0)
rotation_matrix = cv2.getRotationMatrix2D((gr_w//2, gr_h//2), angle=45, scale=1) # angle can be negative (+ is rot to the left, - to the right)
# center - the center of rotation
rotated_gr = cv2.warpAffine(gr, rotation_matrix, (gr_w//2, gr_h//2), borderMode=cv2.INTER_AREA) # applies an affine transformation to an image
# borderMode - pixel interpolation method
# to rotate an image: each pixel (2d-point) multiples by a rot.matrix (2x2) and we get another point
cv2.imshow('45 deg rotated gr image', rotated_gr); cv2.waitKey(0)
tr_matrix = np.array([[1,0, 100],
                    [0, 1, 130]], dtype=np.float32)
cv2.imshow('translated_image', cv2.warpAffine(gr, tr_matrix, (gr.shape[1]//2, gr.shape[0]//2))); cv2.waitKey(0)
cv2.destroyAllWindows()



sys.exit(0)
print(os.path.abspath(os.getcwd()))
print(313*0.3)
k = np.ones((3, 3), np.float32)/25
im = cv2.medianBlur(cv2.filter2D(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB),
                  -1, kernel=k), 5)
#im = cv2.filter2D(cv2.cvtColor(cv2.imread(r"autoencoder_data/train_set/1st-art-oil-painting-reproduction_vertical_and_hor_fl.jpg"), cv2.COLOR_BGR2RGB), kernel=k, ddepth=5)
plt.imshow(im); plt.show()

print(os.listdir('autoencoder_data/train_noise'))