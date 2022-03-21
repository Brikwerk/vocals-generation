# Testing VAE part first

from src.dataset import StemsDataset
from src.vae_gan_model import VAE

import torch

import librosa
import soundfile as sf
from matplotlib import pyplot as plt
import numpy

def rmDim(image):
  if image.shape[0] == 1:
    image = image[0, :, :]
  return image


if __name__ == "__main__":

    # config
    input_size = (1,216,216)
    latent_dim = 128


    model_path = "/content/vocals-generation/model/vae.pth"

    model = VAE(latent_dim = latent_dim, 
        input_size=input_size)
    model.load_state_dict(torch.load(model_path))

    dataset = StemsDataset(
        data_root= "/content/vocals-generation/data",
    )

    dataset_index = 9 
    spec = dataset.__getitem__(dataset_index)['accompaniment'].unsqueeze(0)

    with torch.no_grad():
        output = model(spec)['decoder_out'].squeeze(0)

    output = rmDim(output)
    specDim = rmDim(spec[0])

    # Plot original and reconstructed spectrograms
    fig = plt.figure(figsize=(10, 15), dpi=300)
    ax = fig.add_subplot(211)
    ax.imshow(specDim.cpu().detach().numpy(), aspect='auto', origin='lower')
    ax.set_title('Original')
    ax2 = fig.add_subplot(212)
    ax2.imshow(output.cpu().detach().numpy(), aspect='auto', origin='lower')
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
