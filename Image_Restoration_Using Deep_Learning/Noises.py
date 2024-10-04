import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.util import random_noise


def add_gaussian_noise(image, mean, var):
    # Generate Gaussian noise
    # sigma = var**0.5
    # gaussian_noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
    blurred_image = np.zeros_like(image)
    # Add the Gaussian noise to the image
    for channel in range(blurred_image.shape[-1]):
        blurred_image[:,:,channel] = ((random_noise(image[:,:,channel], mode="gaussian", mean=mean, var=var)) * 255).astype('uint8')
        # blurred_image[:,:,channel] = (gaussian_filter(image[:,:,channel], sigma=1.2)).astype('uint8')

    # noisy_image = cv.add(image, gaussian_noise)
    
    return blurred_image

def add_poisson_noise(image):
    image = image
    noisy_image = image.copy()
    for channel in range(noisy_image.shape[-1]):
        noisy_image[:,:,channel] = ((random_noise(image[:,:,channel], mode="poisson")) * 255).astype('uint8')
    return noisy_image.astype(np.uint8)

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Convert image to float if it is not already
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0
    
    # Create a random matrix
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)
    
    # Add Salt noise
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 1

    # Add Pepper noise
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0

    # Clip values to be in the range [0, 1]
    noisy = np.clip(noisy, 0, 1)
    
    # Convert back to [0, 255] and to uint8 format
    noisy_image = (noisy * 255).astype(np.uint8)
    
    return noisy_image

def add_speckle_noise(image, mean, var):
    # Generate Gaussian noise
    image = image.astype(np.float32) / 255.0
    gaussian_noise = np.random.normal(mean, np.sqrt(var), image.shape)
    
    # Add the Gaussian noise to the image
    noisy_image = image + image * gaussian_noise
    
    # Clip the values to be in the range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)
    
    # Convert back to [0, 255] and to uint8 format
    noisy_image = (noisy_image * 255).astype(np.uint8)
    
    return noisy_image

