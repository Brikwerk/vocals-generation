# Train VAE only 
from src.losses import kl_divergence, gaussian_likelihood 
from src.dataset import StemsDataset
from src.vae_gan_model import VAE

from tqdm import tqdm
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
import os
from matplotlib import pyplot as plt

# for transformations
from torchvision import transforms as T

import numpy as np


def gauss_likelihood(x_decoded, logscale, x):
    return gaussian_likelihood(x_decoded, logscale, x)


def kl_div(z, mu, std):
    return kl_divergence(z, mu, std)


def format_losses(losses):
    if len(losses) > 0:
        return f"Loss: {np.mean(losses):.4f}"
    else:
        return "Loss: None"


if __name__ == "__main__":
    # Config
    input_size = (1,216,216)
    latent_dim = 128
    batch_size = 4
    gpus = 1
    epochs = 10 # originally 550

    spec_type = 'accompaniment'

    out_folder = "./"
    out_path = os.path.join(out_folder, "vae2.pth")

    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # set transformations
    transform = T.Compose([
        T.Normalize(mean=[0.4, 0.4, 0.4],
            std=[.2, 0.2, 0.2]),
    ])

    # Load the dataset
    # Don't apply data augmentation transformations to validation and testing datasets
    dataset = StemsDataset(
        data_root="/home/reece/Data/stem_spectrograms_44khz/",
        
    )
    # add transform=transform later. Does not work right now since not defined in dataset.py

    # split into train/val sets
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #print(train_dataset[0])  # dictionary of 'vocals' and 'accompaniment' tensors!!!
    #print(train_dataset[0]['accompaniment'].shape) # this is torch.Size([1, 216, 216])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    ) 
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    ) 

    # select training device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # instantiate model
    model = VAE(latent_dim = latent_dim, 
        input_size=input_size)
    #device = 'cpu'      #not sure why doing this works
    model = model.to(device)

    # Initialize optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # for loss function
    logscale = nn.Parameter(torch.Tensor([0.0]))

    scaler = torch.cuda.amp.GradScaler()

    # training loop
    output = None
    losses = []
    for epoch in range(epochs):
        with tqdm(total=len(train_loader), unit=" batch") as tepoch:
            for i, batch in enumerate(train_loader):
                tepoch.update(1)

                # Zero out the optimizer
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    x = batch[spec_type]
                    x = x.to(device)
                    output = model(x)

                    decoded_x = output['decoder_out']
                    z = output['encoder_out']['z']
                    mu = output['encoder_out']['mu']
                    logvar = output['encoder_out']['logvar']
                    std = torch.exp(logvar / 2)

                    recon_loss = gauss_likelihood(decoded_x, logscale, x)

                    kl_loss = kl_div(z, mu, std)

                    loss = kl_loss - recon_loss
                    loss = loss.mean()

                    losses.append(loss.item())
                    tepoch.set_postfix(loss=format_losses(losses))

                    if i % 100 == 0:
                        # Plot original and reconstructed spectrograms
                        fig = plt.figure(figsize=(10, 15), dpi=300)
                        ax = fig.add_subplot(211)
                        ax.imshow(x[0].cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
                        ax.set_title('Original')
                        ax2 = fig.add_subplot(212)
                        ax2.imshow(decoded_x[0].cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
                        ax2.set_title('Reconstructed')
                        plt.savefig(os.path.join('./outputs', 'output.png'))
                        plt.close()

                # loss.backward()
                # optimizer.step()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


    torch.save(model.state_dict(), out_path)

  
