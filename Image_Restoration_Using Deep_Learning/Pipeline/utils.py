from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import torch
from ImaData import ImageDataset
from IPython.display import clear_output
import matplotlib.pyplot as plt
import io
import os
from PIL import Image
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Showing an image from the dataset
def show_image():
    dataset = ImageDataset(root_dir=r"D:\Ibrahim_data\train")
    loader = DataLoader(dataset, batch_size=128, num_workers=0)
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    for idx, (low_res, high_res) in enumerate(loader):
        # Display the first image in the left subplot
        axs[0].imshow(low_res[0].permute(1, 2, 0))
        axs[0].set_title("low res")

        # Display the second image in the right subplot
        axs[1].imshow(high_res[0].permute(1, 2, 0))
        axs[1].set_title("high res")
        
        if(idx == 0):
            break
            
    # Show the figure
    plt.show()

# def plot_examples(gen):
#     dataset_test = ImageDataset(root_dir=r"D:\Ibrahim_data\val")
#     loader = DataLoader(dataset_test, batch_size=16, num_workers=0)
    
#     # Create a figure with two subplots
#     fig, axs = plt.subplots(1, 3, figsize=(8, 4))
#     chosen_batch = random.randint(0, len(loader)-1)
#     for idx, (low_res, high_res) in enumerate(loader):
#         if(chosen_batch == idx):
#             chosen = random.randint(0, len(low_res)-1)
        
#             axs[0].set_axis_off()
#             axs[0].imshow(low_res[chosen].permute(1, 2, 0))
#             axs[0].set_title("low res")

#             with torch.no_grad():
#                 upscaled_img = gen(low_res[chosen].to(device).unsqueeze(0))
        
#             axs[1].set_axis_off()
#             axs[1].imshow(upscaled_img.cpu().permute(0, 2, 3, 1)[0])
#             axs[1].set_title("predicted")
        
#             axs[2].set_axis_off()
#             axs[2].imshow(high_res[chosen].permute(1, 2, 0))
#             axs[2].set_title("high res")
        
#             if(idx == 1):
#                 break
            
#     # Show the figure
#     plt.show()      
    
#     gen.train()

def plot_examples(gen, epoch, save_path):
    dataset_test = ImageDataset(root_dir=r"D:\New_Model\Dataset\val")
    loader = DataLoader(dataset_test, batch_size=16, num_workers=0)
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    chosen_batch = random.randint(0, len(loader) - 1)
    
    for idx, (low_res, high_res) in enumerate(loader):
        if chosen_batch == idx:
            chosen = random.randint(0, len(low_res) - 1)
            
            axs[0].set_axis_off()
            axs[0].imshow(low_res[chosen].permute(1, 2, 0))
            axs[0].set_title("low res")
            
            with torch.no_grad():
                upscaled_img = gen(low_res[chosen].to(device).unsqueeze(0))
            
            axs[1].set_axis_off()
            axs[1].imshow(upscaled_img.cpu().permute(0, 2, 3, 1)[0])
            axs[1].set_title("predicted")
            
            axs[2].set_axis_off()
            axs[2].imshow(high_res[chosen].permute(1, 2, 0))
            axs[2].set_title("high res")
            
            if idx == 1:
                break

    # Save the plot to a buffer using plt and convert it to a PIL image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)
    
    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the PIL image at the specified location using the epoch number
    pil_img.save(os.path.join(save_path, f"output_epoch_{epoch}.png"))
    
    # Close the figure to free memory
    plt.close(fig)
    
    gen.train()

def train_progress(epoch, num_epochs, d_losses, g_losses):
    clear_output(wait=True)
    plt.figure(figsize=(10,5))
    plt.title("Training progress")
    plt.plot(d_losses,label="Discriminator loss")
    plt.plot(g_losses,label="Generator loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print(f"Epoch [{epoch}/{num_epochs}], Discriminator loss: {d_losses[-1]:.4f}, Generator loss: {g_losses[-1]:.4f}")

# Create a CSV file only once
def prepare_csv_file(csv_file_path):
    # Check if the file already exists
    if not os.path.exists(csv_file_path):
        # Create the CSV file with headers
        df = pd.DataFrame(columns=["Epoch", "Gen Loss", "Disc Loss", "Val Gen Loss", "Val Disc Loss", "Avg PSNR"])
        df.to_csv(csv_file_path, index=False)
    else:
        print(f"{csv_file_path} already exists, appending to the existing file.")

def save_epoch_results(csv_file_path, epoch, gen_loss, disc_loss, avg_val_gen_loss, avg_val_disc_loss, avg_psnr):
    df = pd.DataFrame({
        "Epoch": [epoch + 1],  # Adding 1 to epoch to start counting from 1
        "Gen Loss": [gen_loss],
        "Disc Loss": [disc_loss],
        "Val Gen Loss": [avg_val_gen_loss],
        "Val Disc Loss": [avg_val_disc_loss],
        "Avg PSNR": [avg_psnr]
    })
    # Append to the CSV file without writing the header again
    df.to_csv(csv_file_path, mode='a', header=False, index=False)


# Create a CSV file only once
def prepare_csv_file_for_val(csv_file_path):
    # Check if the file already exists
    if not os.path.exists(csv_file_path):
        # Create the CSV file with headers
        df = pd.DataFrame(columns=["Epoch","Val Gen Loss", "Val Disc Loss", "PSNR", "SSIM_R", "SSIM_G", "SSIM_B", "SSIM_Full", "Cross_Correlation"])
        df.to_csv(csv_file_path, index=False)
    else:
        print(f"{csv_file_path} already exists, appending to the existing file.")

def save_epoch_results_for_val(csv_file_path, epoch, val_gen_loss, val_disc_loss, psnr, ssim_r, ssim_g, ssim_b, ssim_full, cross_corr):
    df = pd.DataFrame({
        "Epoch": [epoch],  # Adding 1 to epoch to start counting from 1
        "Val Gen Loss": [val_gen_loss],
        "Val Disc Loss": [val_disc_loss],
        "PSNR": [psnr],
        "SSIM_R": [ssim_r],
        "SSIM_G": [ssim_g],
        "SSIM_B": [ssim_b],
        "SSIM_Full": [ssim_full],
        "Cross_Correlation": [cross_corr]    
    })
    # Append to the CSV file without writing the header again
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
