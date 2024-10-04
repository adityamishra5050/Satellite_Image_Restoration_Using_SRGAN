import rasterio
from rasterio.plot import show
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import Noises as ns
import Vignetting as vg
import Chromatic_aberr as ch
import Blurring_Effect as be
import Stripes as st


# Ask the user for the path of the file
Dalian_image_path = input("Please enter the path of the image file: ")

with rasterio.open(Dalian_image_path) as img:
    #show(img)
    data = img.read()

data = np.transpose(data, (1, 2, 0))

# data.shape

# Preparing the plotting function 
def plot_function(Image2, title2):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)  # <-- Correction made here
    plt.imshow(data)
    plt.title('Original Image')
    plt.axis('off')  # Hide the axis
    plt.subplot(1,2,2)  # <-- Correction made here
    plt.imshow(Image2)
    plt.title(title2)
    plt.axis('off')
    plt.show() 

# Ask the user for the noise type and parameters
apply_effect = input("\n choose the effect (Noise/Vignetting/Chromatic/Blur/Striping): ").strip().lower()

if apply_effect == "noise":
#apply_noise_effect = input("Do you want to apply a noise effect? (yes/no): ").strip().lower()
#if apply_noise_effect == "yes":
    noise_type = input("\nEnter the noise type (gaussian/poisson/salt_and_pepper/speckle): ").strip().lower()
    if noise_type == "gaussian":
        mean = input("Enter the mean for Gaussian noise(press Enter to use default 0): ")
        mean = float(mean) if mean else 0
        var = input("Enter the variance for Gaussian noise(press Enter to use default 10): ")
        var = float(var) if var else 10
        Gaus_Noise_img = ns.add_gaussian_noise(data, mean = 0, var = 10)
        plot_function(Gaus_Noise_img, 'Gaussian Noise Image')
    elif noise_type == "poisson":
        poi_noise_image = ns.add_poisson_noise(data)
        plot_function(poi_noise_image, 'Poisson Noise Image')  
    elif noise_type == "salt_and_pepper":
        salt_prob = input("Enter the probability of salt noise(press Enter to use default 0.05): ")
        salt_prob = float(salt_prob) if salt_prob else 0.05
        pepper_prob = input("Enter the probability of pepper noise(press Enter to use default 0.05): ")
        pepper_prob = float(pepper_prob) if pepper_prob else 0.05
        s_n_p_noise_image = ns.add_salt_and_pepper_noise(data, salt_prob, pepper_prob)
        plot_function(s_n_p_noise_image, 'Salt and Pepper Noise Image')      
    elif noise_type == "speckle":
        mean = input("Enter the mean for Speckle noise(press Enter to use default 0): ")
        mean = float(mean) if mean else 0
        var = input("Enter the variance for Speckle noise(press Enter to use default 10): ")
        var = float(var) if var else 10
        spec_noise_image = ns.add_speckle_noise(data, mean, var)
        plot_function(spec_noise_image, 'Speckle Noise Image')
    else:
        print("Invalid noise type entered.")

elif apply_effect == "vignetting":
# Ask the user if they want to apply the vignette effect
#apply_vignette_effect = input("Do you want to apply a vignette effect? (yes/no): ").strip().lower()
#if apply_vignette_effect == "yes":
    sigmaX = input("\nEnter the sigmaX for the vignette effect (press Enter to use default 1900): ")
    sigmaX = float(sigmaX) if sigmaX else 1900
    sigmaY = input("Enter the sigmaY for the vignette effect (press Enter to use default 1800): ")
    sigmaY = float(sigmaY) if sigmaY else 1800
    vignette_image = vg.apply_vignette(data, sigmaX, sigmaY)
    plot_function(vignette_image, 'Image with Vignetting effect')

elif apply_effect == "chromatic":
# Ask the user if they want to apply the vignette effect
#apply_chromaberr_effect = input("Do you want to apply a chromatic aberration effect? (yes/no): ").strip().lower()
#if apply_chromaberr_effect == "yes":
    noise_type = input("\nEnter the noise type (spatial/spherical): ").strip().lower()
    if noise_type == "spatial":
        shiftX = input("Enter the shiftX for the Chromatic Aberration effect (press Enter to use default 100): ")
        shiftX = int(shiftX) if shiftX else 100
        shiftY = input("Enter the shiftY for the chromatic Aberration effect (press Enter to use default 100): ")
        shiftY = int(shiftY) if shiftY else 100
        chrome_abb_image = ch.simulate_chromatic_aberration(data, shift = (shiftX,shiftY))
        plot_function(chrome_abb_image, 'Image with Chromatic Aberration effect')
    elif noise_type == "spherical":
        plot_function(ch.simulate_spherical_chromatic_aberration(data), 'Spherical Chromatic aberration  Image')

elif apply_effect == "blur":
#apply_blurring_effect = input("Do you want to apply a Blurring effect? (yes/no): ").strip().lower()
#if apply_blurring_effect == "yes":
    noise_type = input("\n Enter the noise type (Gaussian/Motion/Radial/Zoom): ").strip().lower()
    if noise_type == "gaussian":
        Gaus_blur_img = be.apply_gaussian_blur(data)
        plot_function(Gaus_blur_img, 'Gaussian Blur Image')
    if noise_type == "motion":
        kernal_size = input("Enter the kernal_size for the Motion Blur Effect (press Enter to use default 15): ")
        kernal_size = int(kernal_size) if kernal_size else 15
        angle = input("Enter the angle for the Motion Blur effect (press Enter to use default 1 degree): ")
        angle = float(angle) if angle else 1
        motion_blur_img = be.apply_motion_blur(data, kernal_size, angle)
        plot_function(motion_blur_img, 'Motion Blur Image')
    if noise_type == "radial":
        radial_blur_img = be.apply_radial_blur(data)
        plot_function(radial_blur_img, 'Radial Blur Image')
    if noise_type == "zoom":
        scale_factor = input("Enter the scale_factor for the Zoom Blur effect (press Enter to use default 0.1): ")
        scale_factor = float(scale_factor) if scale_factor else 0.1
        zoom_blur_img = be.apply_zoom_blur(data, scale_factor)
        plot_function(zoom_blur_img, 'Zoom Blur Image')

elif apply_effect == "striping":
#apply_striping_effect = input("Do you want to apply a Strpping effect? (yes/no): ").strip().lower()
#if apply_striping_effect == "yes":
    blend_factor = input("\n Enter the blend_factor for the Striping effect (press Enter to use default 0.5): ")
    blend_factor = float(blend_factor) if blend_factor else 0.5
    angle = input("Enter the angle for the Striping effect (press Enter to use default 0): ")
    angle = float(angle) if angle else 0
    num_stripes = input("Enter the number of Stripes for the Striping effect (press Enter to use default 100): ")
    num_stripes = int(num_stripes) if num_stripes else 100
    stripe_width = input("Enter the Width of Stripe for the Striping effect (press Enter to use default 2): ")
    stripe_width = int(stripe_width) if stripe_width else 2
    plot_function(st.add_stripping_noise(data, blend_factor , angle, num_stripes, stripe_width), 'Image with Stripping Effect')

else:
    print("Invalid noise type entered.")