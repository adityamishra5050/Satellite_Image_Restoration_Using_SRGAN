import os
import io
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from ImaData import ImageDataset
from Gen import *

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


if __name__ == "__main__":

    def plot_examples(model, save_path):
        dataset_test = ImageDataset(root_dir=r"D:\New_Model\Dataset\test")  # Load dataset
        loader = DataLoader(dataset_test, batch_size=16, num_workers=0)  # DataLoader

        for batch_idx, (low_res, high_res) in enumerate(loader):
            for img_idx in range(len(low_res)):
                # Create a figure with subplots for low-res, predicted, and high-res images
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                
                # Plot Low-Resolution Image
                axs[0].set_axis_off()
                axs[0].imshow(low_res[img_idx].permute(1, 2, 0))
                axs[0].set_title("Low Resolution")
                
                # Generate Upscaled Image (Prediction)
                with torch.no_grad():
                    upscaled_img = model(low_res[img_idx].unsqueeze(0).to(device))
                
                # Plot Predicted Image
                axs[1].set_axis_off()
                axs[1].imshow(upscaled_img.cpu().permute(0, 2, 3, 1)[0])
                axs[1].set_title("Predicted")
                
                # Plot High-Resolution Image
                axs[2].set_axis_off()
                axs[2].imshow(high_res[img_idx].permute(1, 2, 0))
                axs[2].set_title("High Resolution")
                
                # Save the plot for the current image
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(os.path.join(save_path, f"output_batch_{batch_idx}_img_{img_idx}.png"))
                
                # Close the figure to free memory
                plt.close(fig)

    plot_examples(model, save_path=r"D:\New_Model\Prediction_plots\visible_33_plot")
    

# Example usage:
# Assuming `gen` is your pre-trained model and `save_path` is the directory to save images.
# plot_examples(gen, save_path=r"path_to_save_plots")
