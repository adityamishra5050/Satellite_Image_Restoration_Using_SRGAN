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
from PIL import Image


# Ask the user for the path of the file
Dalian_image_path = input("Please enter the path of the image file: ")

with rasterio.open(Dalian_image_path) as img:
    #show(img)
    data = img.read()

data = np.transpose(data, (1, 2, 0))

# data.shape

def save_image_as_tiff(image, variable_name):
    # Convert the NumPy array to an Image object
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Create the file path
    directory = r"C:\Users\Adityamishra\OneDrive - AZISTA INDUSTRIES PVT LTD\Desktop\Aditya_Internship\Azista_Computer_Vision\Image_Restoration_Using Deep_Learning\Distorted_Image_Dataset\Yulin_3_60cm_Distorted_Images"
    file_path = f"{directory}\\{variable_name}.tif"

    # Save the image
    image.save(file_path)
    print(f"Image saved at: {file_path}")

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


Gaus_Noise_img = ns.add_gaussian_noise(data, mean = 0, var = 1.2)
poi_noise_image = ns.add_poisson_noise(data)
s_n_p_noise_image = ns.add_salt_and_pepper_noise(data, salt_prob = 0.05, pepper_prob = 0.05)
spec_noise_image = ns.add_speckle_noise(data, mean = 0, var = 2)
vignette_image = vg.apply_vignette(data, sigmaX = 2200, sigmaY = 2100)
chrome_abb_image = ch.simulate_chromatic_aberration(data, shift = (50,50))
spherical_chro_Image = ch.simulate_spherical_chromatic_aberration(data)
Gaus_blur_img = be.apply_gaussian_blur(data)
motion_blur_img = be.apply_motion_blur(data, kernel_size = 25, angle = 5)
radial_blur_img = be.apply_radial_blur(data)
zoom_blur_img = be.apply_zoom_blur(data, scale_factor = 0.1)
horizontal_stripe_img = st.add_stripping_noise(data, blending_factor = 0.4 , angle = 0, num_stripes = 100 , stripe_width = 2)
vertical_stripe_img = st.add_stripping_noise(data, blending_factor = 0.4, angle = 90, num_stripes = 100 , stripe_width = 2)
diagnol_pos_stripe_img = st.add_stripping_noise(data, blending_factor = 0.4 , angle = 45, num_stripes = 100 , stripe_width = 2)
diagnol_neg_stripe_img = st.add_stripping_noise(data, blending_factor = 0.4 , angle = -45, num_stripes = 100 , stripe_width = 2)

images = {
    "Gaus_Noise_img": Gaus_Noise_img,
    "poi_noise_image": poi_noise_image,
    "s_n_p_noise_image": s_n_p_noise_image,
    "spec_noise_image": spec_noise_image,
    "vignette_image": vignette_image,
    "chrome_abb_image": chrome_abb_image,
    "spherical_chro_Image": spherical_chro_Image,
    "Gaus_blur_img": Gaus_blur_img,
    "motion_blur_img": motion_blur_img,
    "radial_blur_img": radial_blur_img,
    "zoom_blur_img": zoom_blur_img,
    "horizontal_stripe_img": horizontal_stripe_img,
    "vertical_stripe_img": vertical_stripe_img,
    "diagnol_pos_stripe_img": diagnol_pos_stripe_img,
    "diagnol_neg_stripe_img": diagnol_neg_stripe_img
}

# Iterate through the dictionary and save each image
for variable_name, image in images.items():
    save_image_as_tiff(image, variable_name)


