# import os
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# import Noises as ns
# import Vignetting as vg
# import Chromatic_aberr as ch
# import Blurring_Effect as be
# import Stripes as st
# from PIL import Image
# import time
# from tqdm import tqdm

# # Directory containing the original TIFF images
# input_directory = r"C:\Users\Adityamishra\OneDrive - AZISTA INDUSTRIES PVT LTD\Desktop\Aditya_Internship\Azista_Computer_Vision\Image_Restoration_Using Deep_Learning\Image_dataset"

# # Directory to save distorted images
# base_directory = r"C:\Users\Adityamishra\OneDrive - AZISTA INDUSTRIES PVT LTD\Desktop\Aditya_Internship\Azista_Computer_Vision\Image_Restoration_Using Deep_Learning\Final_Distorted_Image_Dataset"

# def save_image_as_tiff(image, variable_name, folder_name):
#     # Convert the NumPy array to an Image object
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)

#     # Create the folder path
#     folder_path = os.path.join(base_directory, folder_name)
#     os.makedirs(folder_path, exist_ok=True)
    
#     # Create the file path
#     file_path = os.path.join(folder_path, f"{variable_name}.tif")

#     # Save the image
#     image.save(file_path)
#     #print(f"Image saved at: {file_path}")

# # Preparing the plotting function 
# def plot_function(original, transformed, title):
#     plt.figure(figsize=(20, 10))
#     plt.subplot(1, 2, 1)
#     plt.imshow(original)
#     plt.title('Original Image')
#     plt.axis('off')  # Hide the axis
#     plt.subplot(1, 2, 2)
#     plt.imshow(transformed)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# # Get a list of TIFF files in the directory
# tiff_files = [f for f in os.listdir(input_directory) if f.endswith('.tif')]

# # Start timing for the entire process
# total_start_time = time.time()

# # Process each TIFF file with a single progress bar
# for tiff_file in tqdm(tiff_files, desc="Processing all images", unit="file"):
#     # Start timing for each individual image
#     start_time = time.time()

#     # Full path of the image file
#     image_path = os.path.join(input_directory, tiff_file)
    
#     with rasterio.open(image_path) as img:
#         data = img.read()

#     # Transpose data if it has more than two dimensions
#     if data.ndim == 3:
#         data = np.transpose(data, (1, 2, 0))
    
#     # Apply transformations
#     Gaus_Noise_img = ns.add_gaussian_noise(data, mean=0, var=1.2)
#     poi_noise_image = ns.add_poisson_noise(data)
#     s_n_p_noise_image = ns.add_salt_and_pepper_noise(data, salt_prob=0.05, pepper_prob=0.05)
#     spec_noise_image = ns.add_speckle_noise(data, mean=0, var=2)
#     vignette_image = vg.apply_vignette(data, sigmaX=2200, sigmaY=2100)
#     chrome_abb_image = ch.simulate_chromatic_aberration(data, shift=(50, 50))
#     spherical_chro_Image = ch.simulate_spherical_chromatic_aberration(data)
#     Gaus_blur_img = be.apply_gaussian_blur(data)
#     motion_blur_img = be.apply_motion_blur(data, kernel_size=25, angle=5)
#     radial_blur_img = be.apply_radial_blur(data)
#     zoom_blur_img = be.apply_zoom_blur(data, scale_factor=0.1)
#     horizontal_stripe_img = st.add_stripping_noise(data, blending_factor=0.4, angle=0, num_stripes=100, stripe_width=2)
#     vertical_stripe_img = st.add_stripping_noise(data, blending_factor=0.4, angle=90, num_stripes=100, stripe_width=2)
#     diagnol_pos_stripe_img = st.add_stripping_noise(data, blending_factor=0.4, angle=45, num_stripes=100, stripe_width=2)
#     diagnol_neg_stripe_img = st.add_stripping_noise(data, blending_factor=0.4, angle=-45, num_stripes=100, stripe_width=2)

#     images = {
#         "Gaus_Noise_img": Gaus_Noise_img,
#         "poi_noise_image": poi_noise_image,
#         "s_n_p_noise_image": s_n_p_noise_image,
#         "spec_noise_image": spec_noise_image,
#         "vignette_image": vignette_image,
#         "chrome_abb_image": chrome_abb_image,
#         "spherical_chro_Image": spherical_chro_Image,
#         "Gaus_blur_img": Gaus_blur_img,
#         "motion_blur_img": motion_blur_img,
#         "radial_blur_img": radial_blur_img,
#         "zoom_blur_img": zoom_blur_img,
#         "horizontal_stripe_img": horizontal_stripe_img,
#         "vertical_stripe_img": vertical_stripe_img,
#         "diagnol_pos_stripe_img": diagnol_pos_stripe_img,
#         "diagnol_neg_stripe_img": diagnol_neg_stripe_img
#     }

#     # Use the image file name (without extension) as the folder name
#     image_name = os.path.splitext(tiff_file)[0]

#     # Iterate through the dictionary and save each image
#     for variable_name, image in images.items():
#         save_image_as_tiff(image, variable_name, image_name)

#     # Calculate and print the time taken for processing the current image
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f"Time taken for {tiff_file}: {time_taken:.2f} seconds")

# # Calculate total processing time
# total_end_time = time.time()
# total_time_taken = total_end_time - total_start_time
# print(f"Total time taken for processing all images: {total_time_taken:.2f} seconds")






# import os
# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# import Noises as ns
# import Vignetting as vg
# import Chromatic_aberr as ch
# import Blurring_Effect as be
# import Stripes as st
# from PIL import Image
# import time
# from tqdm import tqdm
# import random
# from skimage.util import random_noise

# # Directory containing the original TIFF images
# input_directory = r"D:\Image_distortion_datasets_4\Orignal_Image_chips_dataset"

# # Directory to save distorted images
# base_directory = r"D:\Image_distortion_datasets_4\Input_Image"
# # def save_image_as_tiff(image, variable_name, image_name):
# #     # Convert the NumPy array to an Image object
# #     if isinstance(image, np.ndarray):
# #         image = Image.fromarray(image)



# #     # Create the file path
# #     first_part, second_part = image_name.split('_', 1)
# #     file_path = os.path.join(base_directory, f"{first_part}_{variable_name}_{second_part}.tif")

# #     # Save the image
# #     image.save(file_path)
# #     #print(f"Image saved at: {file_path}")

# def save_image_as_tiff(image, variable_name, image_name):
#     # Convert the NumPy array to an Image object
#     if isinstance(image, np.ndarray):
#         image = Image.fromarray(image)



#     # Create the file path
#     first_part, second_part = image_name.split('_', 1)
#     #file_path = os.path.join(base_directory, f"{first_part}_{variable_name}_{second_part}.png")
#     file_path = os.path.join(base_directory, f"{first_part}_{variable_name}_{second_part}.tif")

#     # Save the image
#     image.save(file_path)
#     #print(f"Image saved at: {file_path}")   


# # Get a list of TIFF files in the directory
# tiff_files = [f for f in os.listdir(input_directory) if f.endswith('.tif')]
# #tiff_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

# # Start timing for the entire process
# total_start_time = time.time()

# # Process each TIFF file with a single progress bar
# for tiff_file in tqdm(tiff_files, desc="Processing all images", unit="file"):
#     # Start timing for each individual image
#     # start_time = time.time()

#     # Full path of the image file
#     image_path = os.path.join(input_directory, tiff_file)
    
#     with rasterio.open(image_path) as img:
#         data = img.read()

#     # Transpose data if it has more than two dimensions
#     if data.ndim == 3:
#         data = np.transpose(data, (1, 2, 0))
    
#     # Apply transformations with random parameter selections
#     Gaus_Noise_img = ns.add_gaussian_noise(data, mean=0, var=random.uniform(0.003, 0.006))
#     poi_noise_image = ns.add_poisson_noise(data)
#     #s_n_p_noise_image = ns.add_salt_and_pepper_noise(data, salt_prob=0.05, pepper_prob=0.05)
#     #spec_noise_image = ns.add_speckle_noise(data, mean=0, var=random.uniform(1.0, 2.0))
#     #vignette_image = vg.apply_vignette(data, sigmaX=random.choice([70,80,90,100]), sigmaY=random.choice([70,80,90,100]))
#     #chrome_abb_image = ch.simulate_chromatic_aberration(data, shift=(random.choice([30, 40, 50, 60]), random.choice([30, 40, 50, 60])))
#     #spherical_chro_Image = ch.simulate_spherical_chromatic_aberration(data)
#     Gaus_blur_img = be.apply_gaussian_blur(data)
#     #motion_blur_img = be.apply_motion_blur(data, kernel_size=random.choice([20, 21, 22, 23, 24, 25]), angle=random.choice([2, 3, 4, 5, 6]))
#     #radial_blur_img = be.apply_radial_blur(data)
#     #zoom_blur_img = be.apply_zoom_blur(data, scale_factor=0.1)
#     #horizontal_stripe_img = st.add_stripping_noise(data, blending_factor=random.choice([0.1, 0.2, 0.3, 0.4, 0.5]), angle=0, num_stripes=random.choice([80, 90, 100, 110, 120]), stripe_width=random.choice([2, 3]))
#     #vertical_stripe_img = st.add_stripping_noise(data, blending_factor=random.choice([0.1, 0.2, 0.3, 0.4, 0.5]), angle=90, num_stripes=random.choice([80, 90, 100, 110, 120]), stripe_width=random.choice([2, 3]))
#     #diagnol_pos_stripe_img = st.add_stripping_noise(data, blending_factor=random.choice([0.1, 0.2, 0.3, 0.4, 0.5]), angle=45, num_stripes=random.choice([80, 90, 100, 110, 120]), stripe_width=random.choice([2, 3]))
#     #diagnol_neg_stripe_img = st.add_stripping_noise(data, blending_factor=random.choice([0.1, 0.2, 0.3, 0.4, 0.5]), angle=-45, num_stripes=random.choice([80, 90, 100, 110, 120]), stripe_width=random.choice([2, 3]))

#     images = {
#         "Gaus_Noise_img": Gaus_Noise_img,
#         "poi_noise_image": poi_noise_image,
#         #"s_n_p_noise_image": s_n_p_noise_image,
#         #"spec_noise_image": spec_noise_image,
#         #"vignette_image": vignette_image,
#         #"chrome_abb_image": chrome_abb_image,
#         #"spherical_chro_Image": spherical_chro_Image,
#         "Gaus_blur_img": Gaus_blur_img,
#         #"motion_blur_img": motion_blur_img,
#         #"radial_blur_img": radial_blur_img,
#         #"zoom_blur_img": zoom_blur_img,
#         #"horizontal_stripe_img": horizontal_stripe_img,
#         #"vertical_stripe_img": vertical_stripe_img,
#         #"diagnol_pos_stripe_img": diagnol_pos_stripe_img,
#         #"diagnol_neg_stripe_img": diagnol_neg_stripe_img
#     }

#     # Use the image file name (without extension) as the folder name
#     image_name = os.path.splitext(tiff_file)[0]

#     # Iterate through the dictionary and save each image
#     for variable_name, image in images.items():
#         save_image_as_tiff(image, variable_name, image_name)

#     # Calculate and print the time taken for processing the current image
#     # end_time = time.time()
#     # time_taken = end_time - start_time
#     # print(f"Time taken for {tiff_file}: {time_taken:.2f} seconds")

# # Calculate total processing time
# total_end_time = time.time()
# total_time_taken = total_end_time - total_start_time
# print(f"Total time taken for processing all images: {total_time_taken:.2f} seconds")



import os
import rasterio
import numpy as np
import Noises as ns
import Blurring_Effect as be
from PIL import Image
import time
from tqdm import tqdm
import random
from Dataset_chips import *
from Copy_Images_Creation import *

def process_images_and_save(input_directory):
    """
    Processes images in the input directory by applying transformations and saves them
    in an output folder located in the parent directory of the input folder.
    
    Parameters:
    - input_directory: str, path to the source folder containing TIFF images.

    Returns:
    - output_directory: str, path to the newly created output folder containing processed images.
    """
    
    # Create output folder in the parent directory of the input folder
    parent_directory = os.path.dirname(input_directory)
    output_directory = os.path.join(parent_directory, 'Input_Images')
    os.makedirs(output_directory, exist_ok=True)
    
    def save_image_as_tiff(image, variable_name, image_name, output_directory):
        """
        Saves a given NumPy array as a TIFF image after renaming it.

        Parameters:
        - image: np.ndarray, image data.
        - variable_name: str, name to insert into the filename.
        - image_name: str, base name of the image.
        - output_directory: str, path to save the image.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        first_part, second_part = image_name.split('_', 1)
        file_path = os.path.join(output_directory, f"{first_part}_{variable_name}_{second_part}.tif")
        image.save(file_path)

    # Get list of TIFF files in the input directory
    tiff_files = [f for f in os.listdir(input_directory) if f.endswith('.tif')]

    # Start timing for the entire process
    total_start_time = time.time()

    # Process each TIFF file
    for tiff_file in tqdm(tiff_files, desc="Processing all images", unit="file"):
        image_path = os.path.join(input_directory, tiff_file)

        # Read the image using rasterio
        with rasterio.open(image_path) as img:
            data = img.read()

        # Transpose data if it has more than two dimensions
        if data.ndim == 3:
            data = np.transpose(data, (1, 2, 0))

        # Apply transformations
        Gaus_Noise_img = ns.add_gaussian_noise(data, mean=0, var=random.uniform(0.003, 0.006))
        poi_noise_image = ns.add_poisson_noise(data)
        Gaus_blur_img = be.apply_gaussian_blur(data)

        # Dictionary to store the transformed images
        images = {
            "Gaus_Noise_img": Gaus_Noise_img,
            "poi_noise_image": poi_noise_image,
            "Gaus_blur_img": Gaus_blur_img,
        }

        # Image name without extension
        image_name = os.path.splitext(tiff_file)[0]

        # Save each image
        for variable_name, image in images.items():
            save_image_as_tiff(image, variable_name, image_name, output_directory)

    # Calculate total processing time
    total_end_time = time.time()
    total_time_taken = total_end_time - total_start_time
    print(f"Total time taken for processing all images: {total_time_taken:.2f} seconds")

    # Return the output directory
    return output_directory

# # Define the list of names
# names_list = [
#     "Gaus_Noise_img",
#     "poi_noise_image",
#     "Gaus_blur_img"
# ]


# # Get input from the user
# input_folder = input("Enter the path to the folder containing images: ")
# chip_size = int(input("Enter the chip size (e.g., 512): "))
# # Define source folder
# src_folder = process_images_in_folder(input_folder, chip_size)

# # Execute the function and get the output folder path
# Target_folder = copy_and_rename_images(src_folder, names_list)

# Input_folder =process_images_and_save(Target_folder)


# # Output the result
# print(f"Processed images are saved in: {Input_folder}")