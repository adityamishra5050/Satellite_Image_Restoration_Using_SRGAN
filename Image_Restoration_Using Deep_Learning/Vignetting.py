import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def apply_vignette(image, sigmaX=1900, sigmaY=1800):
    # Read the input image
    rows, cols= image.shape[:2]
    
    # Create Gaussian kernels for X and Y directions
    kernelX = cv.getGaussianKernel(cols, sigmaX)
    kernelY = cv.getGaussianKernel(rows, sigmaY)
    
    # Create the 2D Gaussian kernel by multiplying the X and Y kernels
    kernel = kernelY * kernelX.T
    
    # Normalize the kernel to have values between 0 and 1
    mask = kernel / np.max(kernel)
    
    
    # Create the vignette effect by multiplying each channel of the image by the mask
    vignette = np.copy(image)
    for i in range(3):
        vignette[:, :, i] = vignette[:, :, i] * mask
    
    return vignette
