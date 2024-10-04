
import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def add_stripping_noise(image, blending_factor, angle, num_stripes, stripe_width, intensity_range=(0, 255), orientation='horizontal'):
    # Convert image to float if it is not already
    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    # Create a blank image with double the dimensions of the original image
    height, width, _ = image.shape
    mask_height, mask_width = 2 * height, 2 * width
    noisy_mask = np.zeros((mask_height, mask_width, 3), dtype=np.float32)

    # Determine the number of stripes
    # num_stripes = 500
    
    # Create the stripes
    for _ in range(num_stripes):
        if orientation == 'horizontal':
            y_start = np.random.randint(0, mask_height - stripe_width)
            y_end = y_start + stripe_width
            noisy_mask[y_start:y_end, :, :] = np.random.uniform(intensity_range[0] / 255, intensity_range[1] / 255, (stripe_width, mask_width, 3))
        elif orientation == 'vertical':
            x_start = np.random.randint(0, mask_width - stripe_width)
            x_end = x_start + stripe_width
            noisy_mask[:, x_start:x_end, :] = np.random.uniform(intensity_range[0] / 255, intensity_range[1] / 255, (mask_height, stripe_width, 3))

    # Clip the values to be in the range [0, 1]
    noisy_mask = np.clip(noisy_mask, 0, 1)
    
    # Rotate the mask by -45 degrees
    center = (mask_width // 2, mask_height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotated_mask = cv.warpAffine(noisy_mask, rotation_matrix, (mask_width, mask_height))

    # Create a blank canvas with the same size as the noisy mask
    canvas = np.zeros((mask_height, mask_width, 3), dtype=np.float32)
    
    # Calculate the coordinates to place the original image at the center of the canvas
    y_offset = (mask_height - height) // 2
    x_offset = (mask_width - width) // 2
    
    # Place the original image at the center of the canvas
    canvas[y_offset:y_offset+height, x_offset:x_offset+width, :] = image
    
    # Blend the canvas with the rotated mask using the blending factor
    blended_canvas = canvas + blending_factor * rotated_mask
    
    # Extract the region corresponding to the original image size
    blended_image = blended_canvas[y_offset:y_offset+height, x_offset:x_offset+width, :]
    
    # Convert back to [0, 255] and to uint8 format
    blended_image = (blended_image * 255).astype(np.uint8)
    
    return blended_image

