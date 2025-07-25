import cv2
from PIL import Image
import numpy as np 
import matplotlib, matplotlib.pyplot as plt
import os, sys

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
#sys.exit(0)

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

#sys.exit(0)
# for image in os.listdir(path):
#     im_path = os.path.join(path, image)
#     im = Image.open(im_path); print(im.format, im.size, im.mode) # w, h
#     im_np = np.asarray(im, dtype=np.float32)
#     plt.imshow(im_np); plt.show()
#     print(f'image {im_path}')
#     print(f"np.unit(im_np): {np.uint(im_np)}"); #sys.exit(0)
#     #cv2.imread(im_path); print('\n')


# reshaping images to one dimension - 800, 800
s = (800, 800)

save_dir = r"autoencoder_data/train_set" 
print(len(os.listdir(save_dir)))
# os.path.basename(path): /dir1/dir2/dir3/file.txt -> file.txt (returns filename)
def resize_images(img_paths: str, size: tuple, save_dir: str) -> int: # 0 or 1 if fail/success
    try:
        for img in img_paths: # full paths
            im = cv2.imread(img)
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img)), cv2.resize(im, size))
    except Exception as e:
        print(f"error occured: {e}")
        return 0
    return 1

# save_dir = r"autoencoder_data/train_set" 
# res = resize_images(image_paths, s, save_dir)
# print('\n', f"resized images saved to path {save_dir}" if res else f"an error occured", '\n')

# import glob
# glob.glob(path) ??


preprocessed_images = [os.path.join(save_dir, imp) for imp in os.listdir(save_dir)]
fig = plt.figure()
for i in range(len(save_dir)):
    plt.subplot(5, 6, i+4) # 6 6 i+1
    plt.imshow(cv2.cvtColor(cv2.imread(preprocessed_images[i]), cv2.COLOR_BGR2RGB))
plt.show()

# gaussian, yellowish color noise  (228, 215, 170), (164, 152, 107)
save_dir_noise = save_dir.split('/')[0] + '/train_noise'
print(save_dir_noise)
print(os.path.isabs(save_dir), os.path.isdir(save_dir_noise))
print(os.path.isabs('~/bsh_basics.txt')) # False, why?
# '/home/skrand/bsh_basics.txt' - True
# '~/bsh_basics.txt' - False

