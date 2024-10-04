# import os
# import shutil

# def copy_and_rename_images(src_folder, dest_folder, names_list):
#     # Ensure destination folder exists
#     os.makedirs(dest_folder, exist_ok=True)

#     # Iterate over all files in the source folder
#     for filename in os.listdir(src_folder):
#         if filename.endswith(".tiff") or filename.endswith(".tif"):
#             base_name, ext = os.path.splitext(filename)
#             parts = base_name.split('_', 1)  # Split the base name at the first underscore

#             if len(parts) > 1:
#                 first_part, remaining_parts = parts
#                 for name in names_list:
#                     # Construct new filename by inserting the name after the first underscore
#                     new_filename = f"{first_part}_{name}_{remaining_parts}{ext}"
                    
#                     # Define source and destination paths
#                     src_path = os.path.join(src_folder, filename)
#                     dest_path = os.path.join(dest_folder, new_filename)
                    
#                     # Copy and rename the file
#                     shutil.copy(src_path, dest_path)
#                     print(f"Copied and renamed: {src_path} -> {dest_path}")

# # def copy_and_rename_images(src_folder, dest_folder, names_list):
# #     # Ensure destination folder exists
# #     os.makedirs(dest_folder, exist_ok=True)

# #     # Iterate over all files in the source folder
# #     for filename in os.listdir(src_folder):
# #         if filename.endswith(".png") :
# #             base_name, ext = os.path.splitext(filename)
# #             parts = base_name.split('_', 1)  # Split the base name at the first underscore

# #             if len(parts) > 1:
# #                 first_part, remaining_parts = parts
# #                 for name in names_list:
# #                     # Construct new filename by inserting the name after the first underscore
# #                     new_filename = f"{first_part}_{name}_{remaining_parts}{ext}"
                    
# #                     # Define source and destination paths
# #                     src_path = os.path.join(src_folder, filename)
# #                     dest_path = os.path.join(dest_folder, new_filename)
                    
# #                     # Copy and rename the file
# #                     shutil.copy(src_path, dest_path)
# #                     print(f"Copied and renamed: {src_path} -> {dest_path}")                    

# # Define the list of names
# names_list = [
#     "Gaus_Noise_img",
#     "poi_noise_image",
#     #"s_n_p_noise_image",
#     #"spec_noise_image",
#     #"vignette_image",
#     #"chrome_abb_image",
#     #"spherical_chro_Image",
#     "Gaus_blur_img",
#     #"motion_blur_img",
#     #"radial_blur_img",
#     # "zoom_blur_img",
#     # "horizontal_stripe_img",
#     # "vertical_stripe_img",
#     # "diagnol_pos_stripe_img",
#     # "diagnol_neg_stripe_img"
# ]

# # Define source and destination folders
# src_folder = r"D:\Image_distortion_datasets_Madhapur\Orignal_Image_chips_dataset_1"  # Replace with your source folder path
# dest_folder = r"D:\Image_distortion_datasets_Madhapur\Target_Image_1"   # Replace with your destination folder path

# # Execute the function
# copy_and_rename_images(src_folder, dest_folder, names_list)

import os
import shutil
from Dataset_chips import *

def copy_and_rename_images(src_folder, names_list):
    """
    Copies and renames images from the source folder to a new output folder created in the parent directory.

    Parameters:
    - src_folder: str, path to the source folder containing the images.
    - names_list: list, list of names to be inserted into the renamed files.

    Returns:
    - dest_folder: str, path to the destination folder where renamed files are saved.
    """
    # Get the parent directory of the source folder
    parent_dir = os.path.dirname(src_folder)
    
    # Create the output folder in the parent directory
    dest_folder = os.path.join(parent_dir, 'Target_Images')
    os.makedirs(dest_folder, exist_ok=True)

    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            base_name, ext = os.path.splitext(filename)
            parts = base_name.split('_', 1)  # Split the base name at the first underscore

            if len(parts) > 1:
                first_part, remaining_parts = parts
                for name in names_list:
                    # Construct new filename by inserting the name after the first underscore
                    new_filename = f"{first_part}_{name}_{remaining_parts}{ext}"
                    
                    # Define source and destination paths
                    src_path = os.path.join(src_folder, filename)
                    dest_path = os.path.join(dest_folder, new_filename)
                    
                    # Copy and rename the file
                    shutil.copy(src_path, dest_path)
                    #print(f"Copied and renamed: {src_path} -> {dest_path}")

    # Return the destination folder path
    return dest_folder

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
# output_folder = copy_and_rename_images(src_folder, names_list)

# # Output the result
# print(f"Renamed images are saved in: {output_folder}")

 