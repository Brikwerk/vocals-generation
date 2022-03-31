from asyncio.log import logger
import random

from src.dataset import StemsDataset
from src.vae_models import LitVAE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import numpy as np


if __name__ == "__main__":
    # Config
    input_size = (1, 216, 216)
    encoder_output_dim = 512 # Resnet18
    # encoder_output_dim = 2048 # Resnet50
    latent_dim = 1024
    batch_size = 3
    gpus = 1
    epochs = 300

    # Ensure reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Create an amend encoder and decoder
    # for processing spectrograms.
    encoder = resnet18_encoder(False, False)
    encoder.conv1 = nn.Conv2d(input_size[0], 64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    decoder = resnet18_decoder(
        latent_dim=latent_dim,
        input_height = input_size[1],
        first_conv=False,
        maxpool1=False
    )
    decoder.conv1 = nn.Conv2d(64, input_size[0],  # ResNet18
    # decoder.conv1 = nn.Conv2d(256, input_size[0],  # ResNet50
        kernel_size=(3, 3),
        stride=(1, 1), 
        padding=(1, 1), 
        bias=False
    )

    # Load the dataset
    dataset = StemsDataset(
        data_root= r'C:\Users\kafkacat\Desktop\EltonOut',
    )

    # Split into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
    )

    vae = LitVAE(encoder, decoder,
        enc_output_dim=encoder_output_dim,
        latent_dim=latent_dim,
        input_size=input_size)
    tb_logger = pl_loggers.TensorBoardLogger("./logs/", "VAE")
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        logger=tb_logger,
        resume_from_checkpoint = r"C:\Users\kafkacat\Desktop\vocals-generation\logs\VAE\version_12\checkpoints\epoch=298-step=275858.ckpt")
    trainer.fit(vae, train_loader)
