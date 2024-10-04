import os
import shutil
from sklearn.model_selection import train_test_split
from Dataset_chips import *
from Copy_Images_Creation import *
from Full_Data_Distortion import *

def split_images_into_folders(folder_low_res, folder_high_res):
    """
    Splits images from two input folders (low_res and high_res) into train, validation, and test sets, 
    and saves them in an output folder within the parent directory of folder_low_res.

    Args:
        folder_low_res (str): Path to the folder containing low-resolution images.
        folder_high_res (str): Path to the folder containing high-resolution images.

    Returns:
        str: The path of the created output folder.
    """

    # Use the parent directory of 'folder_low_res' to create the output folder
    parent_dir = os.path.dirname(folder_low_res)
    output_folder = os.path.join(parent_dir, "Dataset")

    # Define the paths for the train, test, and validation splits inside the output folder
    output_dirs = {
        'train': {'low_res': os.path.join(output_folder, 'train', 'low_res'), 'high_res': os.path.join(output_folder, 'train', 'high_res')},
        'val': {'low_res': os.path.join(output_folder, 'val', 'low_res'), 'high_res': os.path.join(output_folder, 'val', 'high_res')},
        'test': {'low_res': os.path.join(output_folder, 'test', 'low_res'), 'high_res': os.path.join(output_folder, 'test', 'high_res')}
    }

    # Create the output directories if they don't exist     
    for split in output_dirs.values():
        for path in split.values():
            os.makedirs(path, exist_ok=True)

    # Get the list of images (assuming both folders have the same images with identical names)
    images = os.listdir(folder_low_res)

    # Split the images into train, test, and validation sets
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
    train_images, val_images = train_test_split(train_images, test_size=0.20, random_state=42)  # 0.20 x 0.8 = 0.16

    # Function to copy images to respective folders
    def copy_images(images, src_folder, dest_folder):
        for img in images:
            src_path = os.path.join(src_folder, img)
            dest_path = os.path.join(dest_folder, img)
            shutil.copy(src_path, dest_path)

    # Copy images for Folder low_res (Input)
    copy_images(train_images, folder_low_res, output_dirs['train']['low_res'])
    copy_images(val_images, folder_low_res, output_dirs['val']['low_res'])
    copy_images(test_images, folder_low_res, output_dirs['test']['low_res'])

    # Copy images for Folder high_res (Target)
    copy_images(train_images, folder_high_res, output_dirs['train']['high_res'])
    copy_images(val_images, folder_high_res, output_dirs['val']['high_res'])
    copy_images(test_images, folder_high_res, output_dirs['test']['high_res'])

    print(f"Images have been successfully split into train, test, and validation sets for both 'low_res' and 'high_res' folders.")
    print(f"Output folder: {output_folder}")

    # Return the path of the output folder
    return output_folder

# Define the list of names
names_list = [
    "Gaus_Noise_img",
    "poi_noise_image",
    "Gaus_blur_img"
]


# Get input from the user
input_folder = input("Enter the path to the folder containing images: ")
chip_size = int(input("Enter the chip size (e.g., 512): "))
# Define source folder
src_folder = process_images_in_folder(input_folder, chip_size)

# Execute the function and get the output folder path
Target_folder = copy_and_rename_images(src_folder, names_list)

Input_folder = process_images_and_save(src_folder)

Dataset_folder = split_images_into_folders(Input_folder, Target_folder)

# Output the result
print(f"Your Dataset is Preapared inside the folder : {Dataset_folder}")