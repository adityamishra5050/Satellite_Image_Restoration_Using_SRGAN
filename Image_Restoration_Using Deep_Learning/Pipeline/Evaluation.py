import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
from torchvision.models import vgg19
import os
from Gen import *
from ImaData import ImageDataset
from torch.utils.data import DataLoader
import math
import torch.nn.functional as F
#from skimage.metrics import structural_similarity as ssim
import cv2
from scipy.signal import correlate2d
from utils import  save_epoch_results_for_val , prepare_csv_file_for_val
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim


if __name__ == "__main__":
#show_image()
    def calculate_psnr(img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(1.0 / math.sqrt(mse.item()))
    
    def calculate_ssim_per_band(img1, img2, C1=1e-6, C2=1e-6):
        # Calculate SSIM for each band
        bands_ssim = []
        for i in range(img1.shape[0]):  # Assuming img1 and img2 are [C, H, W]
            mu1 = F.avg_pool2d(img1[i:i+1], kernel_size=3, stride=1, padding=1)
            mu2 = F.avg_pool2d(img2[i:i+1], kernel_size=3, stride=1, padding=1)

            sigma1 = F.avg_pool2d(img1[i:i+1] * img1[i:i+1], kernel_size=3, stride=1, padding=1) - mu1 * mu1
            sigma2 = F.avg_pool2d(img2[i:i+1] * img2[i:i+1], kernel_size=3, stride=1, padding=1) - mu2 * mu2
            sigma12 = F.avg_pool2d(img1[i:i+1] * img2[i:i+1], kernel_size=3, stride=1, padding=1) - mu1 * mu2

            ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            ssim_d = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2)
            ssim_index = ssim_n / ssim_d

            bands_ssim.append(ssim_index.mean().item())  # Average SSIM for the band

        return bands_ssim

    def calculate_ssim_full_image(img1, img2, C1=1e-6, C2=1e-6):
        # Calculate SSIM for the full image
        mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
        mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

        sigma1 = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1 * mu1
        sigma2 = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2 * mu2
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1 * mu2

        ssim_n = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        ssim_d = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1 + sigma2 + C2)
        ssim_index = ssim_n / ssim_d

        return ssim_index.mean().item()
    
    def calculate_cross_correlation(img1, img2):
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)
        correlation = F.cosine_similarity(img1_flat, img2_flat, dim=0)
        return correlation.item()

    
    csv_file_path = r"D:\New_Model\Saved_Result\validation_epoch_results_1.csv"
    prepare_csv_file_for_val(csv_file_path)

    val = ImageDataset(root_dir=r"D:\New_Model\Dataset\val")
    val_loader = DataLoader(val, batch_size=16, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen =  Generator(in_channels=3).to(device)
    disc = Discriminator(in_channels=3).to(device)
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = vggL()
    # Create an instance of the model
    model = gen 

    num_epochs = 120

    # Loop through your epochs
    for epoch in range(1, num_epochs + 1):
        #loop = tqdm(val_loader)
        num_images = 0
        psnr_total = 0
        ssim_b_total = 0 
        ssim_g_total = 0
        ssim_r_total = 0
        ssim_full_total = 0
        cross_corr_total = 0
        num_batch = 0 
        val_gen_loss_total = 0
        val_disc_loss_total = 0
        num_val_batches = 0

        # Construct checkpoint file names
        gen_checkpoint_file = f'D:/New_Model/All_checkpoints/checkpoint_{epoch}_gen.pt' 
        disc_checkpoint_file = f'D:/New_Model/All_checkpoints/checkpoint_{epoch}_disc.pt'
        # Load the model from the checkpoints
        model.load_state_dict(torch.load(gen_checkpoint_file))
        disc.load_state_dict(torch.load(disc_checkpoint_file))

         # If you are using the model for inference, set it to evaluation mode
        model.eval()
        disc.eval()

        with torch.no_grad():
            for idx, (low_res_val, high_res_val) in enumerate(val_loader):
                high_res_val = high_res_val.to(device)
                low_res_val = low_res_val.to(device)

                # Generate fake images
                fake_val = gen(low_res_val)

                # Validation discriminator loss
                disc_real_val = disc(high_res_val)
                disc_fake_val = disc(fake_val.detach())

                disc_loss_real_val = bce(disc_real_val, torch.ones_like(disc_real_val))
                disc_loss_fake_val = bce(disc_fake_val, torch.zeros_like(disc_fake_val))
                val_disc_loss = disc_loss_fake_val + disc_loss_real_val
                val_gen_loss_total += val_disc_loss

                # Validation generator loss
                disc_fake_val = disc(fake_val)
                adversarial_loss_val = 1e-3 * bce(disc_fake_val, torch.ones_like(disc_fake_val))
                loss_for_vgg_val = 0.006 * vgg_loss(fake_val, high_res_val)
                val_gen_loss = loss_for_vgg_val + adversarial_loss_val
                val_disc_loss_total += val_gen_loss

                # Calculate PSNR
                for i in range(len(high_res_val)):
                    psnr_value = calculate_psnr(fake_val[i], high_res_val[i])
                    psnr_total += psnr_value
                    ssim = calculate_ssim_per_band(fake_val[i], high_res_val[i])
                    ssim_b, ssim_g, ssim_r = ssim[0], ssim[1], ssim[2]
                    ssim_b_total += ssim_b
                    ssim_g_total += ssim_g
                    ssim_r_total += ssim_r
                    ssim_full = calculate_ssim_full_image(fake_val[i], high_res_val[i])
                    ssim_full_total += ssim_full
                    cross_corr = calculate_cross_correlation(fake_val[i], high_res_val[i])
                    cross_corr_total += cross_corr

                    num_images += 1

                num_val_batches += 1


            avg_psnr = psnr_total / num_images
            avg_SSIM_R = ssim_b_total / num_images
            avg_SSIM_G = ssim_g_total / num_images
            avg_SSIM_B = ssim_r_total / num_images
            avg_ssim_full = ssim_full_total / num_images
            avg_cross_corr = cross_corr_total / num_images

        avg_val_gen_loss = val_gen_loss_total / num_val_batches
        avg_val_disc_loss = val_disc_loss_total / num_val_batches



        save_epoch_results_for_val(csv_file_path, epoch,avg_val_gen_loss.item(), avg_val_disc_loss.item(), avg_psnr, avg_SSIM_R, avg_SSIM_G, avg_SSIM_B, avg_ssim_full, avg_cross_corr)










