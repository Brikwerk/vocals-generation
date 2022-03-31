from os.path import join

from src.losses import kl_divergence, gaussian_likelihood

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchaudio
import pytorch_lightning as pl
from matplotlib import pyplot as plt


class LitVAE(pl.LightningModule):
    def __init__(self, encoder, decoder, spec_type='accompaniment',
                 enc_output_dim=512, latent_dim=1024,       #ENC OUPUT DIM WAS 2048
                 input_size=(4, 284, 283), outputs_dir="./outputs"):
        super().__init__()

        self.outputs_dir = outputs_dir

        self.spec_type = spec_type

        self.encoder = encoder
        self.decoder = decoder

        self.pool_resize = nn.AdaptiveAvgPool2d(input_size[1:])

        self.mu_fc = nn.Linear(enc_output_dim, latent_dim)
        self.logvar_fc = nn.Linear(enc_output_dim, latent_dim)
        self.logscale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def gauss_likelihood(self, x_decoded, logscale, x):
        return gaussian_likelihood(x_decoded, logscale, x)

    def kl_div(self, z, mu, std):
        return kl_divergence(z, mu, std)

    # Prediction/inference
    def forward(self, x):
        encoded_x = self.encoder(x)

        mu = self.mu_fc(encoded_x)
        logvar = self.logvar_fc(encoded_x)

        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        x = batch[self.spec_type]
        encoded_x = self.encoder(x)

        mu = self.mu_fc(encoded_x)
        logvar = self.logvar_fc(encoded_x)

        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        decoded_x = self.decoder(z)
        # decoded_x = self.pool_resize(decoded_x)

        if batch_idx % 300 == 0:
            # save_spec_tensor(decoded_x[0], 'output.wav')
            
            try:
                # Plot original and reconstructed spectrograms
                fig = plt.figure(figsize=(10, 15), dpi=300)
                ax = fig.add_subplot(211)
                ax.imshow(x[0].cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
                ax.set_title('Original')
                ax2 = fig.add_subplot(212)
                ax2.imshow(decoded_x[0].cpu().permute(1,2,0).detach().numpy(), aspect='auto', origin='lower')
                ax2.set_title('Reconstructed')
                plt.savefig(join(self.outputs_dir, 'output.png'))
                plt.close()
            except MemoryError:
                pass


        recon_loss = self.gauss_likelihood(decoded_x, self.logscale, x)

        kl_loss = self.kl_div(z, mu, std)

        loss = kl_loss - recon_loss
        loss = loss.mean()

        self.log_dict({
            'train_loss': loss.item(),
            'recon_loss': recon_loss.mean().item(),
            'kl_loss': kl_loss.mean().item(),
        })

        return loss


class SpecEncoder(nn.Module):
    def __init__(self, latent_dim=128, input_size=(4, 1025, 862)):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding='same'),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding='same'),
            nn.MaxPool2d(4),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        # Calculate the size of the linear layer
        with torch.no_grad():
            x = self.encoder(torch.zeros(input_size).unsqueeze(0))
            self.feature_size = x.shape
            self.linear_size = self.flatten(x).shape[1]
            print(self.feature_size, self.linear_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self.linear_size, latent_dim * 16),
            nn.ReLU(),
            nn.Linear(latent_dim * 16, latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8, latent_dim * 4),
        )

        self.mu = nn.Linear(latent_dim * 4, latent_dim)
        self.logvar = nn.Linear(latent_dim * 4, latent_dim)

    def reparameterize(self, mu, logvar):
        batch = mu.size(0)
        dim = mu.size(1)
        epsilon = torch.randn(batch, dim)
        return mu + torch.exp(logvar / 2) * epsilon

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)

        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterize(mu, logvar)

        return {
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


class SpecDecoder(nn.Module):
    def __init__(self, latent_dim=128, linear_out=26624, reshape_size=(16, 13),
                 output_size=(4, 1025, 862)):
        super().__init__()

        self.reshape_size = reshape_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 8),
            nn.ReLU(),
            nn.Linear(latent_dim * 8, linear_out),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(128, 64, 2, padding='same'),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.Conv2d(64, 64, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(64, 32, 2, padding='same'),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.Conv2d(32, 32, 3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            nn.Conv2d(32, 16, 2, padding='same'),
            nn.Conv2d(16, 16, 3, padding='same'),
            nn.Conv2d(16, 16, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding='same'),
            nn.Tanh(),
            nn.AdaptiveAvgPool2d(output_size[1:]),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.reshape_size[0], self.reshape_size[1])
        x = self.decoder(x)
        return x


class SpecVAE(nn.Module):
    def __init__(self, latent_dim=128, input_size=(4, 1025, 862)):
        super().__init__()

        self.encoder = SpecEncoder(latent_dim, input_size)
        self.decoder = SpecDecoder(latent_dim, self.encoder.linear_size, 
                                   self.encoder.feature_size[2:], input_size)

    def forward(self, x):
        encoder_out = self.encoder(x)
        z = encoder_out['z']
        decoder_out = self.decoder(z)
        return {
            'decoder_out': decoder_out,
            'encoder_out': encoder_out
        }


if __name__ == "__main__":
    # Encoder
    spec_encoder = SpecEncoder()
    print(spec_encoder)
    x = torch.randn(1, 4, 1025, 862)
    x = spec_encoder(x)
    print(x)

    # Decoder
    spec_decoder = SpecDecoder()
    print(spec_decoder)
    x = torch.randn(1, 128)
    x = spec_decoder(x)
    print(x.shape)

    # VAE
    spec_vae = SpecVAE()
    print(spec_vae)
    x = torch.randn(1, 4, 1025, 862, dtype=torch.float32)
    x = spec_vae(x)
    print(x['decoder_out'].shape)

    # Transform decoder output into a complex spectrogram
    output = x['decoder_out']
    spec = torch.randn(1, 2, 1025, 862, dtype=torch.cfloat)
    spec.real = output[:, :2, :, :]
    spec.imag = output[:, 2:, :, :]
    spec = spec.squeeze(0)
    print(spec.shape)

    # Save random decoded waveform
    # ispec_transform = torchaudio.transforms.InverseSpectrogram(n_fft=2048)
    # waveform = ispec_transform(spec, 20 * 44100)
    # torchaudio.save('test.wav', waveform, 44100)
    
