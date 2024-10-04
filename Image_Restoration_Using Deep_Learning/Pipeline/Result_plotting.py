import pandas as pd
import matplotlib.pyplot as plt

def plot_training_data(csv_file_path, csv_file_path_1 , save_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)
    data_1 = pd.read_csv(csv_file_path_1)

    # Extract data for plotting
    epochs = data['Epoch']
    gen_loss_train = data['Gen Loss']
    gen_loss_val = data_1['Val Gen Loss']
    disc_loss_train = data['Disc Loss']
    disc_loss_val = data_1['Val Disc Loss']
    psnr_val = data_1['PSNR']

    # Plot 1: Generator Loss vs Epoch for Training and Validation
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, gen_loss_train, label='Training', color='orange')
    plt.plot(epochs, gen_loss_val, label='Validation', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.title('Generator Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/Gen_loss_vs_epoch.png')
    plt.close()

    # Plot 2: Discriminator Loss vs Epoch for Training and Validation
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, disc_loss_train, label='Training', color='orange')
    plt.plot(epochs, disc_loss_val, label='Validation', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/Disc_loss_vs_epoch.png')
    plt.close()

    # Plot 3: PSNR vs Epoch (Validation only)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_val, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (Validation)')
    plt.title('PSNR vs Epoch (Validation)')
    plt.grid(True)
    plt.savefig(f'{save_path}/psnr_vs_epoch.png')
    plt.close()

# Example usage
plot_training_data(r"D:\Saved_results\epoch_results.csv", r"D:\Saved_results\validation_epoch_results_1.csv" ,r"D:\Saved_results")
