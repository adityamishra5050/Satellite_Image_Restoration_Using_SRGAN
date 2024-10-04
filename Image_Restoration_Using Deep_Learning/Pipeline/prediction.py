import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import os
from Gen import *


if __name__ == "__main__":
#show_image()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen =  Generator(in_channels=3).to(device)
    # Create an instance of the model
    model = gen

    # Define the path to the weight file
    weight_file_path = r"D:\New_Model\All_checkpoints\checkpoint_33_gen.pt"

    # Load the state dictionary (weights)
    model.load_state_dict(torch.load(weight_file_path))

    # If you are using the model for inference, set it to evaluation mode
    model.eval()

    def predict_image(image_path, model):
        """
        Loads an image, applies transformations, passes it through the model, and displays the results.

        Parameters:
        - image_path (str): Path to the image to be processed.
        - model (torch.nn.Module): Trained PyTorch model for prediction.
        """
        # Load the image using Pillow
        input_image = np.array(Image.open(image_path))
        
        low_res_size = 128
        
        #print("Aditya")
        
        transform_low = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((low_res_size, low_res_size)),
        transforms.ToTensor(),
        ])
        
        #print("Aditya1")

        # Apply the transform_low transformations (resize, and convert to tensor)
        input_image_tensor = transform_low(input_image[:, :, :3])

        # Add batch dimension [1, C, H, W] (since the model expects a batch)
        input_image_tensor = input_image_tensor.unsqueeze(0)

        # Move the tensor to GPU if CUDA is available
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model = model.to(device)
        input_image_tensor = input_image_tensor.to(device)
        
        #print("Aditya2")

        # Forward pass through the model
        with torch.no_grad():
            output = model(input_image_tensor)

        # Post-process the output: remove batch dimension and convert back to image format
        output_image = output.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
        output_image = np.moveaxis(output_image, 0, -1)  # Change shape to [H, W, C]

        # Normalize the output image for display (optional)
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())

        return output_image



    def predict_and_save_images(input_folder, output_folder, model):
        """
        Processes all images in the input folder, applies the model prediction, and saves the output images to the output folder.

        Parameters:
        - input_folder (str): Path to the folder containing input images.
        - output_folder (str): Path to the folder where predicted images will be saved.
        - model (torch.nn.Module): Trained PyTorch model for prediction.
        """
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over all files in the input folder
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(input_folder, filename)
                #print(f'Processing {image_path}...')

                # Predict the image using the provided model
                predicted_image = predict_image(image_path, model)
                predicted_image_pil = Image.fromarray((predicted_image * 255).astype(np.uint8))

                #print("Aditya")

                # Save the predicted image to the output folder
                output_image_path = os.path.join(output_folder, filename)
                predicted_image_pil.save(output_image_path)
                #print(f'Saved predicted image to {output_image_path}')


    # Example usage:
    # Assuming `model` is your trained PyTorch model
    input_folder = r"D:\New_Model\Dataset\test\low_res"  # Replace with your input folder path
    output_folder = r"D:\New_Model\Predicted_Images\visible_33_epoch" # Replace with your output folder path

    predict_and_save_images(input_folder, output_folder, model)


