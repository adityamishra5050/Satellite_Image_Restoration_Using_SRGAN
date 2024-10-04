import torch
from tqdm import tqdm
import math
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to calculate PSNR
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

# def train_fn(train_loader, disc, gen, opt_gen, opt_disc, bce, vgg_loss):
#     loop = tqdm(train_loader)
#     disc_loss = 0
#     gen_loss = 0

#     for idx, (low_res, high_res) in enumerate(loop):
#         high_res = high_res.to(device)
#         low_res = low_res.to(device)
        
#         ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
#         fake = gen(low_res)
        
#         disc_real = disc(high_res)
#         disc_fake = disc(fake.detach())
        
#         disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
#         disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        
#         disc_loss = disc_loss_fake + disc_loss_real

#         opt_disc.zero_grad()
#         disc_loss.backward()
#         opt_disc.step()

#         # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
#         disc_fake = disc(fake)
#         adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
#         loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
#         gen_loss = loss_for_vgg + adversarial_loss

#         opt_gen.zero_grad()
#         gen_loss.backward()
#         opt_gen.step()

            
#     return gen_loss.detach().cpu(), disc_loss.detach().cpu()

def train_fn(train_loader,val_loader, disc, gen, opt_gen, opt_disc, bce, vgg_loss):
    loop = tqdm(train_loader)
    disc_loss = 0
    gen_loss = 0
    val_gen_loss = 0
    val_disc_loss = 0
    psnr_total = 0
    num_val_batches = 0
    num_images = 0

    val_gen_loss_total = 0
    val_disc_loss_total = 0

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(device)
        #print(high_res.shape)
        low_res = low_res.to(device)
        #print(low_res.shape)
        
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        fake = gen(low_res)
        
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        
        disc_loss_real = bce(disc_real, torch.ones_like(disc_real))
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        
        disc_loss = disc_loss_fake + disc_loss_real

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        loss_for_vgg = 0.006 * vgg_loss(fake, high_res)
        gen_loss = loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

    # Validation loop
    gen.eval()  # Set the generator to evaluation mode
    disc.eval()  # Set the discriminator to evaluation mode
    
    with torch.no_grad():
        for idx, (low_res_val, high_res_val) in enumerate(val_loader):
            high_res_val = high_res_val.to(device)
            
            low_res_val = low_res_val.to(device)
            
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
                num_images += 1
            
            num_val_batches += 1

    avg_psnr = psnr_total / num_images
    avg_val_gen_loss = val_gen_loss_total / num_val_batches
    avg_val_disc_loss = val_disc_loss_total / num_val_batches

    gen.train()  # Set the generator back to training mode
    disc.train()  # Set the discriminator back to training mode


            
    return gen_loss.detach().cpu(), disc_loss.detach().cpu(),avg_val_gen_loss.cpu(), avg_val_disc_loss.cpu(), avg_psnr