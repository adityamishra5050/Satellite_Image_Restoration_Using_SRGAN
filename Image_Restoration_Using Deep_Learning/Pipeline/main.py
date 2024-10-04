from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.models import vgg19
import torch
from torch import optim
from utils import show_image, plot_examples, save_epoch_results, prepare_csv_file
from Gen import *
from ImaData import ImageDataset
from train import train_fn


#from prediction import predict_image, predict_and_save_images

if __name__ == "__main__":
#show_image()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    lr = 3e-4
    epochs = 120
    batch_size = 16
    num_workers = 0
    img_channels = 3

    # listing the model to take the required subset of it
    test_vgg_model = vgg19(weights=True).eval().to(device)
    lf = list(test_vgg_model.features)
    # lf[25]

    # define the generator / discriminator / and other hyperparameters (not already defined above)
    gen = Generator(in_channels=3).to(device)
    disc = Discriminator(in_channels=3).to(device)
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.9, 0.999))
    #mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = vggL()

    # the data loaders for training and validation
    train = ImageDataset(root_dir=r"D:\Ibrahim_data\train")
    train_loader = DataLoader(train, batch_size=batch_size, num_workers=0, pin_memory=False)

    val = ImageDataset(root_dir=r"D:\Ibrahim_data\val")
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=0)

    d_losses = []
    g_losses = []
    d_val_losses = []
    g_val_losses = []
    av_psnr = []

    csv_file_path = r"D:\Saved_results\epoch_results.csv"
    prepare_csv_file(csv_file_path)
    
    for epoch in range(epochs):
        plot_examples(gen, epoch, save_path=r"D:\epoch_result")
        print("epoch ", epoch+1, "/", epochs)
        gen_loss, disc_loss, avg_val_gen_loss, avg_val_disc_loss, avg_psnr = train_fn(train_loader, val_loader, disc, gen, opt_gen, opt_disc, bce, vgg_loss)
        print(f"Training Generator Loss: {gen_loss}")
        print(f"Training Discriminator Loss: {disc_loss}")
        print(f"Validation Generator Loss: {avg_val_gen_loss}")
        print(f"Validation Discriminator Loss: {avg_val_disc_loss}")
        print(f"Average PSNR: {avg_psnr}")
        # train discriminator and generator and update losses
        d_losses.append(disc_loss)
        g_losses.append(gen_loss)
        d_val_losses.append(d_val_losses)
        g_val_losses.append(g_val_losses)
        av_psnr.append(av_psnr)

        save_epoch_results(csv_file_path, epoch, gen_loss.item(), disc_loss.item(), avg_val_gen_loss.item(), avg_val_disc_loss.item(), avg_psnr)


        torch.save(gen.state_dict(), f"D:/SRGAN_Model/checkpoint_{epoch + 1}_gen.pt")
        torch.save(disc.state_dict(), f"D:/SRGAN_Model/checkpoint_{epoch + 1}_disc.pt")



    # torch.save(gen.state_dict(), r"D:\SRGAN_Model\checkpoint_gen.pt")
    # torch.save(disc.state_dict(), r"D:\SRGAN_Model\checkpoint_disc.pt")

