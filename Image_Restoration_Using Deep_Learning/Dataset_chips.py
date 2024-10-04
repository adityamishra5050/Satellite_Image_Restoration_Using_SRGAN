# import rasterio
# import numpy as np
# import os
# from glob import glob

# def pad_image_to_multiple_of_size(image_path, chip_size):
#     """
#     Pads an image to the nearest multiple of the given chip size in both dimensions.

#     Parameters:
#     - image_path: str, path to the image file.
#     - chip_size: int, the size of the chip for both width and height.

#     Returns:
#     - padded_image: np.ndarray, the padded image array.
#     - profile: dict, the updated profile for the padded image.
#     """
#     # Open the image using rasterio
#     with rasterio.open(image_path) as src:
#         image = src.read()  # Read the image as a NumPy array
#         profile = src.profile

#     # Get the current dimensions
#     original_height, original_width = image.shape[1], image.shape[2]

#     # Calculate residuals
#     residual_height = original_height % chip_size
#     residual_width = original_width % chip_size

#     # Calculate the required padding to make dimensions multiples of chip_size
#     pad_bottom = (chip_size - residual_height) % chip_size
#     pad_right = (chip_size - residual_width) % chip_size

#     # Apply padding to the right and bottom
#     padded_image = np.pad(
#         image,
#         ((0, 0), (0, pad_bottom), (0, pad_right)),
#         mode='constant',
#         constant_values=0
#     )

#     # Verify new dimensions
#     new_height, new_width = padded_image.shape[1], padded_image.shape[2]
#     assert new_height % chip_size == 0 and new_width % chip_size == 0, "The padded dimensions are not multiples of chip size"

#     # Update profile with new dimensions
#     profile.update(height=new_height, width=new_width)

#     return padded_image, profile

# def create_chips(image, image_name, output_dir, chip_size):
#     """
#     Creates image chips of the specified size from a padded image and saves them.

#     Parameters:
#     - image: np.ndarray, the padded image array.
#     - image_name: str, base name of the image to use for naming chips.
#     - output_dir: str, directory where chips will be saved.
#     - chip_size: int, the size of each chip for both width and height.
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Get the dimensions of the image
#     channels, height, width = image.shape

#     # Iterate over the image to create chips
#     for i in range(0, height, chip_size):
#         for j in range(0, width, chip_size):
#             # Extract the chip
#             chip = image[:, i:i+chip_size, j:j+chip_size]

#             # Create the chip filename
#             chip_filename = f"{image_name}_{i//chip_size}_{j//chip_size}.tif"
#             #chip_filename = f"{image_name}_{i//chip_size}_{j//chip_size}.png"

#             # Save the chip using rasterio
#             chip_path = os.path.join(output_dir, chip_filename)
#             with rasterio.open(chip_path, 'w', driver='GTiff', 
#                                height=chip.shape[1], width=chip.shape[2], 
#                                count=chip.shape[0], dtype=chip.dtype) as dst:
#                 dst.write(chip)
#             # with rasterio.open(chip_path, 'w', driver='PNG', 
#             #                    height=chip.shape[1], width=chip.shape[2], 
#             #                    count=chip.shape[0], dtype=chip.dtype) as dst:
#             #     dst.write(chip)

#             print(f"Saved chip: {chip_path}")

# def process_images_in_folder(input_folder, output_folder, chip_size):
#     """
#     Processes all images in the input folder, pads them, and creates chips.

#     Parameters:
#     - input_folder: str, directory containing the input images.
#     - output_folder: str, directory where output chips will be saved.
#     - chip_size: int, the size of each chip for both width and height.
#     """
#     # Find all image files in the folder (adjust the extension if needed)
#     image_files = glob(os.path.join(input_folder, '*.tif'))
#     # image_files = glob(os.path.join(input_folder, '*.png'))


#     for image_file in image_files:
#         print(f"Processing: {image_file}")

#         # Get the base name of the image without extension
#         base_name = os.path.splitext(os.path.basename(image_file))[0]

#         # Pad the image
#         padded_image, padded_profile = pad_image_to_multiple_of_size(image_file, chip_size)

#         # Create chips for the image
#         create_chips(padded_image, base_name, output_folder, chip_size)

# # Get chip size from the user
# chip_size = int(input("Enter the chip size (e.g., 512): "))

# # Paths to your input and output directories

# input_folder = r"D:\Image_distortion_datasets_4\Original_Image_Dataset"
# output_folder = r"D:\Image_distortion_datasets_4\Orignal_Image_chips_dataset"

# # Process all images in the folder
# process_images_in_folder(input_folder, output_folder, chip_size)





import rasterio
import numpy as np
import os
from glob import glob

def pad_image_to_multiple_of_size(image_path, chip_size):
    """
    Pads an image to the nearest multiple of the given chip size in both dimensions.

    Parameters:
    - image_path: str, path to the image file.
    - chip_size: int, the size of the chip for both width and height.

    Returns:
    - padded_image: np.ndarray, the padded image array.
    - profile: dict, the updated profile for the padded image.
    """
    # Open the image using rasterio
    with rasterio.open(image_path) as src:
        image = src.read()  # Read the image as a NumPy array
        profile = src.profile

    # Get the current dimensions
    original_height, original_width = image.shape[1], image.shape[2]

    # Calculate residuals
    residual_height = original_height % chip_size
    residual_width = original_width % chip_size

    # Calculate the required padding to make dimensions multiples of chip_size
    pad_bottom = (chip_size - residual_height) % chip_size
    pad_right = (chip_size - residual_width) % chip_size

    # Apply padding to the right and bottom
    padded_image = np.pad(
        image,
        ((0, 0), (0, pad_bottom), (0, pad_right)),
        mode='constant',
        constant_values=0
    )

    # Verify new dimensions
    new_height, new_width = padded_image.shape[1], padded_image.shape[2]
    assert new_height % chip_size == 0 and new_width % chip_size == 0, "The padded dimensions are not multiples of chip size"

    # Update profile with new dimensions
    profile.update(height=new_height, width=new_width)

    return padded_image, profile

def create_chips(image, image_name, output_dir, chip_size):
    """
    Creates image chips of the specified size from a padded image and saves them.

    Parameters:
    - image: np.ndarray, the padded image array.
    - image_name: str, base name of the image to use for naming chips.
    - output_dir: str, directory where chips will be saved.
    - chip_size: int, the size of each chip for both width and height.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the dimensions of the image
    channels, height, width = image.shape

    # Iterate over the image to create chips
    for i in range(0, height, chip_size):
        for j in range(0, width, chip_size):
            # Extract the chip
            chip = image[:, i:i+chip_size, j:j+chip_size]

            # Create the chip filename
            chip_filename = f"{image_name}_{i//chip_size}_{j//chip_size}.tif"

            # Save the chip using rasterio
            chip_path = os.path.join(output_dir, chip_filename)
            with rasterio.open(chip_path, 'w', driver='GTiff', 
                               height=chip.shape[1], width=chip.shape[2], 
                               count=chip.shape[0], dtype=chip.dtype) as dst:
                dst.write(chip)

            print(f"Saved chip: {chip_path}")

def process_images_in_folder(input_folder, chip_size):
    """
    Processes all images in the input folder, pads them, and creates chips.

    Parameters:
    - input_folder: str, directory containing the input images.
    - chip_size: int, the size of each chip for both width and height.

    Returns:
    - output_folder: str, path to the output folder where chips are saved.
    """
    # Find all image files in the folder (adjust the extension if needed)
    image_files = glob(os.path.join(input_folder, '*.tif'))

    # Get the parent directory of the input folder
    parent_dir = os.path.dirname(input_folder)

    # Define the output folder path inside the parent directory
    output_folder = os.path.join(parent_dir, 'Chipped_Images')

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for image_file in image_files:
        print(f"Processing: {image_file}")

        # Get the base name of the image without extension
        base_name = os.path.splitext(os.path.basename(image_file))[0]

        # Pad the image
        padded_image, padded_profile = pad_image_to_multiple_of_size(image_file, chip_size)

        # Create chips for the image
        create_chips(padded_image, base_name, output_folder, chip_size)

    return output_folder

# # Get input from the user
# input_folder = input("Enter the path to the folder containing images: ")
# chip_size = int(input("Enter the chip size (e.g., 512): "))

# Process all images in the folder and return the output folder path
# output_folder = process_images_in_folder(input_folder, chip_size)

# Output the result
#print(f"Chipped images are saved in: {output_folder}")
