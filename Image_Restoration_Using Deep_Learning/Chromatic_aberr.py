import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def simulate_chromatic_aberration(image,shift):
    # Split the image into its BGR channels
    # Create a copy of the image

    noisy_image = np.copy(image)
    b, g, r = cv.split(noisy_image)

    # Apply a shift to each channel
    r_shifted = np.roll(r, shift[0], axis=0)
    r_shifted = np.roll(r_shifted, shift[1], axis=1)

    g_shifted = np.roll(g, shift[0]//2, axis=0)
    g_shifted = np.roll(g_shifted, shift[1]//2, axis=1)

    b_shifted = np.roll(b, shift[0]//3, axis=0)
    b_shifted = np.roll(b_shifted, shift[1]//3, axis=1)

    # Merge the channels back into a BGR image
    aberrated_image = cv.merge((b_shifted, g_shifted, r_shifted))

    return aberrated_image

def simulate_spherical_chromatic_aberration(image, shift_factors=(100, 50, 150)):
    """
    Simulate Spherical Chromatic Aberration by applying radial shifts to color channels.
    
    Parameters:
    - image: The input image in BGR format.
    - shift_factors: A tuple containing the shift factors for the Blue, Green, and Red channels.
    
    Returns:
    - aberrated_image: The image with simulated spherical chromatic aberration.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Create an empty image for the aberrated result
    aberrated_image = np.zeros_like(image)
    
    # Split the image into its BGR channels
    b, g, r = cv.split(image)
    
    # Create a grid of coordinates
    x, y = np.indices((height, width))
    
    # Calculate the distance from the center

    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Apply radial shift based on distance from center
    b_shifted = np.zeros_like(b)
    g_shifted = np.zeros_like(g)
    r_shifted = np.zeros_like(r)
    
    for i in range(height):
        for j in range(width):
            rad = int(radius[i, j])
            b_x = min(width - 1, max(0, j + int(shift_factors[0] * rad / max(radius.shape))))
            b_y = min(height - 1, max(0, i + int(shift_factors[0] * rad / max(radius.shape))))
            g_x = min(width - 1, max(0, j + int(shift_factors[1] * rad / max(radius.shape))))
            g_y = min(height - 1, max(0, i + int(shift_factors[1] * rad / max(radius.shape))))
            r_x = min(width - 1, max(0, j + int(shift_factors[2] * rad / max(radius.shape))))
            r_y = min(height - 1, max(0, i + int(shift_factors[2] * rad / max(radius.shape))))
            
            aberrated_image[i, j, 0] = b[b_y, b_x]
            aberrated_image[i, j, 1] = g[g_y, g_x]
            aberrated_image[i, j, 2] = r[r_y, r_x]
    
    return aberrated_image