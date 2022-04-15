from dataset import StemsDataset
from .src.vae_models import LitVAE

import torch
import torch.nn.functional as F
import torch.nn as nn
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
import librosa
import soundfile as sf
from matplotlib import pyplot as plt


if __name__ == "__main__":
    model_path = "FOLDER_PATH_TO_MODEL_GOES_HERE"

    input_size = (1, 216, 216)
    # encoder_output_dim = 512 # Resnet18
    encoder_output_dim = 2048 # Resnet50
    latent_dim = 1024
    encoder = resnet50_encoder(False, False)
    encoder.conv1 = nn.Conv2d(input_size[0], 64,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        bias=False
    )
    decoder = resnet50_decoder(
        latent_dim=latent_dim,
        input_height = input_size[1],
        first_conv=False,
        maxpool1=False
    )
    # decoder.conv1 = nn.Conv2d(64, input_size[0],  # ResNet18
    decoder.conv1 = nn.Conv2d(256, input_size[0],  # ResNet50
        kernel_size=(3, 3),
        stride=(1, 1), 
        padding=(1, 1), 
        bias=False
    )

    model = LitVAE.load_from_checkpoint(model_path, encoder=encoder, decoder=decoder)

    dataset = StemsDataset(
        data_root='FOLDER_PATH_TO_DATA_GOES_HERE',
    )

    dataset_index = 1369
    spec = dataset.__getitem__(dataset_index)['accompaniment'].unsqueeze(0)

    with torch.no_grad():
        output = model(spec).squeeze(0)

    # Plot original and reconstructed spectrograms
    fig = plt.figure(figsize=(10, 15), dpi=300)
    ax = fig.add_subplot(211)
    ax.imshow(spec[0].cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
    ax.set_title('Original')
    ax2 = fig.add_subplot(212)
    ax2.imshow(output.cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
    ax2.set_title('Reconstructed')
    plt.savefig(f'./outputs/output_inf.png')
    plt.close()

    output = spec.numpy().squeeze()

    S_dB = (output - 1) * 80
    sr = 22100
    hop_length = 1024

    # Convert S_dB back to power
    S = librosa.db_to_power(S_dB)

    # Convert mel-spectrogram to audio
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr,
        hop_length=hop_length)

    # Save audio
    sf.write('./outputs/orig_inf.wav', y, sr)
