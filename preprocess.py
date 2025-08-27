import cv2
from PIL import Image
import numpy as np 
import matplotlib, matplotlib.pyplot as plt
import os, sys
import random as r

# get clean data, add (custom) noise to it, crop and train 
# paired images have a direct damaged-to-undamaged mapping and unpaired images do not


# where to store env vars (HOME etc) for paths...?
# .env files are usually stored outside project's git files...like in some local file which doesn't get commited or VPS? vercel .env file etc

HOME = os.environ['HOME']
print(os.getcwd(), HOME)
path = os.path.join(HOME, r".cache/kagglehub/datasets/AI_for_Art_Restoration_2/unpaired_dataset_art/undamaged")
print(f"we have: {len(os.listdir(path))} images in the '{path.split('/')[-1]}' folder")

# unsigned - non-negative 
# sigend - negative + non-negative
print(b:='ufloat' in dir(np)) # true
#print(dir(np)) # methods and attributes in numpy module

# RGB, CMYK (cyan(blue) magneta(purple) yellow key(black - key color))

# get min, average, max image sizes 
# os.path.join(os.listdir(path, i_path))
# w,h 
image_paths = [os.path.join(path, ip) for ip in os.listdir(path)]
widths = [Image.open(p).size[0] for p in image_paths]
heights = [Image.open(p).size[1] for p in image_paths]
min_w = min(widths)
min_h = min(heights)
max_w = max(widths)
max_h = max(heights)
avg_w = sum(widths) / len(widths)
avg_h = sum(heights) / len(heights)
print(f"min_w: {min_w}, min_h: {min_h}, avg_w: {avg_w:.2f}, avg_h: {avg_h:.2f}, max_w: {max_w}, max_h: {max_h}")
# image dimensions frequency bar chart

# category: <= (400, 400) <= (800, 800) <= (1200, 1200) <= (1600, 1400) <= (2047, 1600)
numbers = []
a, b, c, d, e = 0, 0, 0, 0, 0
for w, h in zip(widths, heights):
    if w <= 400 and h <= 400:
        a += 1; continue
    if w <= 800 and h <= 800:
        b += 1; continue
    if 1200 <= 800 and h <= 1200:
        c += 1; continue
    if w <= 1600 and h <= 1400:
        d += 1; continue
    if w <= 2047 and h <= 1600:
        e += 1; continue

numbers.extend([a, b, c, d, e])
print(numbers)
categories = ['<= (400, 400)', '<= (800, 800)', '<= (1200, 1200)', '<= (1600, 1400)', '<= (2047, 1600)']
plt.xlabel('Image sizes by groups'); plt.ylabel('Image count')
plt.bar(categories, numbers)
plt.show()

# for image in os.listdir(path):
#     im_path = os.path.join(path, image)
#     im = Image.open(im_path); print(im.format, im.size, im.mode) # w, h
#     im_np = np.asarray(im, dtype=np.float32)
#     plt.imshow(im_np); plt.show()
#     print(f'image {im_path}')
#     print(f"np.unit(im_np): {np.uint(im_np)}"); #sys.exit(0)
#     #cv2.imread(im_path); print('\n')

# reshaping images to one dimension - 800, 800
#s = (800, 800)

save_dir = r"autoencoder_data/train_orig" 
print(len(os.listdir(save_dir)))
# os.path.basename(path): /dir1/dir2/dir3/file.txt -> file.txt (returns filename)
def preprocess(img_paths: str, size: tuple, save_dir: str) -> int: # 0 or 1 fail/success
    try:
        for img in img_paths: # full paths
            resized_im = cv2.resize(cv2.imread(img, cv2.IMREAD_UNCHANGED), size)
            base = os.path.splitext(os.path.join(save_dir, os.path.basename(img)))[0]
            cv2.imwrite(base + '.jpg', resized_im)
            cv2.imwrite(base + '_vertical_fl.jpg', cv2.flip(resized_im, 0))
            cv2.imwrite(base + '_vertical_and_hor_fl.jpg', cv2.flip(resized_im, 1))
            cv2.imwrite(base + '_hor_fl.jpg', cv2.flip(resized_im, -1))
        return 1
    except Exception as e:
        print(f"error occured: {e}")
        return 0

# save_dir = r"autoencoder_data/train_orig" 
# res = preprocess(image_paths, s, save_dir)
# print('\n', f"resized images saved to path {save_dir}" if res else f"an error occured", '\n')

# import glob
# glob.glob(path)


def plot_images(pth):
    assert os.path.isdir(pth) and len(os.listdir(pth))!=0, 'not a dir; or no images in dir'
    abp = os.path.abspath(pth)
    print(f"abp: {abp}")
    for i, absp_im in enumerate(os.listdir(abp)[:25]):
        im_p = os.path.join(abp, absp_im)
        plt.subplot(5,5, i+1) # <=25 images max
        # i+1 - which cell to use (index starts from 1) so i+2 would be 2nd cell for 1st image...
        img = cv2.cvtColor(cv2.imread(im_p), cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()


# p = 'autoencoder_data/train_orig'
# plot_images(p)

# gaussian
# yellowish color noise 
# sepia effect
# adding noise should be random (like for image1 it's Gaussian and for image90 it's yellow)
save_dir_noise = save_dir.split('/')[0] + '/train_noise'
print(save_dir_noise)
print(os.path.isabs(save_dir), os.path.isdir(save_dir_noise))
print(os.path.isabs('~/bsh_basics.txt')) # False, why?


def sepia(src_image):
    sepia_im = np.full_like(src_image, 1, dtype=np.float32) 
    
    #solid color
    sepia_im[:,:,0] *= 20
    sepia_im[:,:,1] *= 66 
    sepia_im[:,:,2] *= 112 

    src_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(src_gray, np.float32)/255 

    #hadamard product
    sepia_im[:,:,0] *= normalized_gray # 2d * 2d
    sepia_im[:,:,1] *= normalized_gray 
    sepia_im[:,:,2] *= normalized_gray 
    return sepia_im.astype(np.uint8)
    


# Gaussian + yellowish + sepia effect
def add_noise(train_original_path, train_noise_path, third_path, historical_color: list[int,int,int], k_size: int, size: tuple) -> int:
    '''
    Function to add noise to images in folder `train_original_path`

    the second number is the portion of images having a yellowish shade (for paintings)

    `color` - tuple with RGB values, for the yellowish shade to make images look 'older'

    `k` - kernel size for adding Gaussian blur 

    Returns 1 if the operation is successful and 0 otherwise
    '''
    assert type(historical_color)==list and len(historical_color)==3 and all([type(n)==int for n in historical_color]) \
            and all([0<=n<=255 for n in historical_color]), "'color' parameter should be a tuple of 3 int values"
    try:
        original_ims = os.listdir(train_original_path)
        print(f"in function 'add_noise' len(original_ims): {len(original_ims)}")
        k = np.ones((k_size, k_size), dtype=np.float32)/25
        yellow_np = np.zeros(size, dtype=np.uint8)
        yellow_np[:, :, :3] = historical_color
        yellow_np[:, :, 3] = 100  
        for oim in original_ims:
            # image_name.jpg
            print(f"image {oim}: \n")
            im = cv2.imread(os.path.join(train_original_path, oim), cv2.IMREAD_UNCHANGED)
            im_name = oim.split('.jpg')[0]
            #cv2.imwrite(os.path.join(third_path, im_name+'_.jpg'), im.astype(np.uint8)) 
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            gaussian_im = cv2.medianBlur(cv2.filter2D(im, -1, kernel=k), 5)  
            gim_p = im_name + '_gaussian'  + '_.jpg'
            cv2.imwrite(os.path.join(third_path, gim_p), im.astype(np.uint8)) 
            cv2.imwrite(os.path.join(train_noise_path, gim_p), gaussian_im.astype(np.uint8))

            # save the same image (im1 -> im1_gauss, im1 -> im1_sepia, ...)
            
            print('\n')
            
            yellowish_im = cv2.addWeighted(im, 0.7, yellow_np, 0.3, 0)
            yellowish_p = im_name + '_yellowish' + '_.jpg'
            cv2.imwrite(os.path.join(third_path, yellowish_p), im.astype(np.uint8))
            cv2.imwrite(os.path.join(train_noise_path, yellowish_p), yellowish_im.astype(np.uint8))

            sepia_im = sepia(im)
            sepia_p = im_name + '_sepia'  + '_.jpg'
            cv2.imwrite(os.path.join(third_path, sepia_p), im.astype(np.uint8))
            cv2.imwrite(os.path.join(train_noise_path, sepia_p), sepia_im.astype(np.uint8))
        return 1
    
    except Exception as e: 
        print(f"error: {e}\n")
        return 0


#print(len([im for im in os.listdir(p) if im.endswith('.jpg')]))

# train_original_path = r"autoencoder_data/train_orig"
# original_ims = [os.path.join(os.path.abspath(train_original_path), im) for im in os.listdir(os.path.abspath(train_original_path))]
# print(original_ims[:2])
# cv2.imwrite(r"autoencoder_data/train_noise/new_.jpg", cv2.imread(r'autoencoder_data/train_orig/9_hor_fl.jpg', cv2.IMREAD_UNCHANGED))

# im = cv2.imread(r"autoencoder_data/train_orig/2a097cb79465dc62db47ff6ae6a20a2a_hor_fl.jpg")
# print(im.shape)

TRAIN_ORIG_PATH = r"autoencoder_data/train_orig"
TRAIN_NOISE_PATH = r"autoencoder_data/train_noise"
THIRD_P = r"autoencoder_data/train_orig_2"
kernel_size = 5
image_shape=(800, 800, 4)
res = add_noise(TRAIN_ORIG_PATH, TRAIN_NOISE_PATH, THIRD_P, [225, 193, 110][::-1], kernel_size, image_shape)  
print(res)

print(len(os.listdir(TRAIN_ORIG_PATH)), len(os.listdir(TRAIN_NOISE_PATH)), len(os.listdir(THIRD_P)))
print(os.listdir(TRAIN_NOISE_PATH)[:3], os.listdir(THIRD_P)[:3])

print(all([noise_name == orig_name for noise_name, orig_name in zip(os.listdir(TRAIN_NOISE_PATH), os.listdir(THIRD_P))]))

for i, (orig_im, noise_im) in enumerate(zip(os.listdir(THIRD_P)[:4], os.listdir(TRAIN_NOISE_PATH)[:4])):
    plt.subplot(2, 4, i+1)  
    orig_img = cv2.cvtColor(cv2.imread(os.path.join(THIRD_P, orig_im)), cv2.COLOR_BGR2RGB)
    plt.imshow(orig_img); plt.title(f'orig_img {i+1}')
    plt.subplot(2, 4, i+5)  
    noise_img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN_NOISE_PATH, noise_im)), cv2.COLOR_BGR2RGB)
    plt.imshow(noise_img); plt.title(f'noise_img {i+1}')
plt.show()

# 372/4 = 94 - -original, not-augmented images
# gaussian (372) + yellowish (372) + sepia (372)
# 372*3 = should be 1116 images with noise in total
# + images in the other folders (w/wo noise, from the kaggle dataset)

