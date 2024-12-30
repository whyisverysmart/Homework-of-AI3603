import os
from PIL import Image

source_imgs_dir='./Data/cats_crop/'
target_imgs_dir='./Data/resized/'
count = 0
for file in os.listdir(source_imgs_dir):
    im = Image.open(source_imgs_dir + file)
    out = im.resize((64, 64), Image.LANCZOS)
    out.save(target_imgs_dir + file)
    count += 1
    if count % 500 == 0:
        print(count, 'images resized')
