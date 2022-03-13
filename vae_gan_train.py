from asyncio.log import logger
#import random

from src.dataset import StemsDataset
from src.vae_gan_model import VAEGAN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# for transformations
from torchvision import transforms as T

import numpy as np


if __name__ == "__main__":
    # Config
    input_size = (1,216,216)
    latent_dim = 128
    batch_size = 4
    gpus = 1
    epochs = 2 # originally 550

    spec_type = 'accompaniment'

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
        data_root=FOLDER_PATH_TO_DATA_GOES_HERE,
        
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
    model = VAEGAN(latent_dim = latent_dim, 
        input_size=input_size)
    model.to(device)

    # tb_logger = pl_loggers.TensorBoardLogger("./logs/", "VAE")

    # Initialize optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Initialise loss function? (already set in VAEGAN)
    #vaegan_loss = model.loss()

    # training loop
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            # Zero out the optimizer
            optimizer.zero_grad()

            output = model(batch[spec_type])

            dis_out = output['discriminator_out']
            decoder_out = output['decoder_out']
            encoder_out = output['encoder_out']

            loss = model.loss(encoder_out, dis_out)
            loss.backward()
            optimizer.step()
