import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.util import random_noise

def apply_gaussian_blur(data) :
    #data = cv.cvtColor(data,cv.COLOR_BGR2RGB)
    
    # Gaussian Blur 
    Gaussian = cv.GaussianBlur(data, (5, 5), 0.8) 
    #Gaussian = cv.cvtColor(Gaussian,cv.COLOR_BGR2RGB)
    
    return Gaussian

def apply_motion_blur(data, kernel_size, angle):
    """
    Apply motion blur to an image.

    Parameters:
    - data: Input image as a NumPy array.
    - kernel_size: Size of the motion blur kernel.
    - angle: Angle of the motion blur in degrees.

    Returns:
    - blurred_image: Output image with motion blur applied.
    """
    #data = cv.cvtColor(data,cv.COLOR_BGR2RGB) 
    # cv.imshow('Original Image', data)
    # cv.waitKey(0)

    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)  #kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size): Sets the middle row of the kernel to ones.
    kernel = kernel / kernel_size #Normalizes the kernel by dividing all elements by the kernel size, making the sum of the kernel elements equal to 1.

    # Rotate the kernel to the specified angle
    rotation_matrix = cv.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle, 1) #Generates a rotation matrix to rotate the kernel around its center by the specified angle.
    kernel = cv.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size)) #Applies the rotation matrix to the kernel, effectively rotating the motion blur kernel to the desired angle.

    # Apply the motion blur kernel to the image
    blurred_image = cv.filter2D(data, -1, kernel) #Applies the motion blur kernel to the input image using convolution. The -1 indicates that the output image will have the same depth as the input image.
    # blurred_image = cv.cvtColor(blurred_image,cv.COLOR_BGR2RGB)
    # cv.imwrite('Motion Blur Image.tiff',blurred_image)
    # cv.imshow('Motion Blurring', blurred_image)
    # cv.waitKey(0)

    return blurred_image

def apply_radial_blur(data, num_blurs=2, angle_step=1):
    """
    Apply radial blur to an image.

    Parameters:
    - data: Input image as a NumPy array.
    - num_blurs: Number of blur layers to blend.
    - angle_step: Angle step in degrees for each blur layer.

    Returns:
    - blurred_image: Output image with radial blur applied.
    """
    #data = cv.cvtColor(data,cv.COLOR_BGR2RGB)
    h, w = data.shape[:2]
    center = (w // 2, h // 2)
    
    # Create an empty image to accumulate the blurs
    accumulated_blur = np.zeros_like(data, dtype=np.float32)
    
    for i in range(num_blurs):
        # Rotate the image slightly
        rotation_matrix = cv.getRotationMatrix2D(center, i * angle_step, 1)
        rotated_image = cv.warpAffine(data, rotation_matrix, (w, h))

        # Accumulate the rotated images
        accumulated_blur += rotated_image.astype(np.float32)
    
    # Normalize the accumulated image
    blurred_image = cv.convertScaleAbs(accumulated_blur / num_blurs)
    
    return blurred_image

def apply_zoom_blur(data, scale_factor, num_levels=1):
    """
    Apply zoom blur to an image.

    Parameters:
    - data: Input image as a NumPy array.
    - num_levels: Number of zoom levels to blend.
    - scale_factor: Factor by which the image is zoomed in at each level.

    Returns:
    - blurred_image: Output image with zoom blur applied.
    """
    h, w = data.shape[:2]
    center = (w // 2, h // 2)
    
    # Create an empty image to accumulate the blurs
    accumulated_blur = np.zeros_like(data, dtype=np.float32)
    
    for i in range(num_levels):
        # Calculate the scale for this level
        scale = scale_factor ** i
        
        # Scale the image
        scaled_image = cv.resize(data, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
        
        # Crop the scaled image to the original size centered around the center of the scaled image
        center_scaled = (scaled_image.shape[1] // 2, scaled_image.shape[0] // 2)
        x1 = center_scaled[0] - w // 2
        y1 = center_scaled[1] - h // 2
        x2 = x1 + w
        y2 = y1 + h
        cropped_image = scaled_image[y1:y2, x1:x2]
        
        # Accumulate the cropped images
        accumulated_blur += cropped_image.astype(np.float32)
    
    # Normalize the accumulated image
    blurred_image = cv.convertScaleAbs(accumulated_blur / num_levels)

    return blurred_image