# image_resizer.py
# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 512
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    print(path)
    if(path != 'dataset/.DS_Store'):
        image = Image.open(path).convert('RGB').resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

        training_data.append(np.asarray(image))

# training_data = np.reshape(
#     training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
# training_data = training_data / 127.5 - 1

training_data = np.reshape(
    np.array(training_data), (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 511.5 - 1

print('saving file...')
print(training_data.shape)
np.save('cyanotype_data_large.npy', training_data)
