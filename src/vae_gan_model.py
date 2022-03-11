import torch
import torch.nn as nn
from torch.autograd import Variable

class VGEncoder(nn.Module):
    def __init__(self, latent_dim=128, input_size=(4, 226, 226)):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding='same'),
            nn.MaxPool2d(3), #maxpool brings size down??!?!?!
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.flatten = nn.Flatten()
        # Calculate the size of the linear layer
        with torch.no_grad():
            x = self.encoder(torch.zeros(input_size).unsqueeze(0))
            self.feature_size = x.shape
            self.linear_size = self.flatten(x).shape[1]
            print(self.feature_size, self.linear_size)
        
        self.fc = nn.Sequential(
            nn.Linear(self.linear_size, latent_dim*16), 
            nn.ReLU(),
        )
        
        # two linear to get the mu vector and the diagonal of the log_variance
        self.mu = nn.Linear(latent_dim * 16, latent_dim)
        self.logvar = nn.Linear(latent_dim * 16, latent_dim)

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

class VGDecoder(nn.Module):
    def __init__(self, latent_dim=128, linear_out=16384, reshape_size=(16,8)):
        super().__init__()

        self.reshape_size = reshape_size
        
        self.fc = nn.Sequential( 
            nn.Linear(latent_dim, linear_out),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv2d(latent_dim, 256, 5, padding='same'), #replace first param with 128? Same thing, but hardcode vs flex?
            nn.ReLU(),
            nn.BatchNorm2d(256), 
            nn.Upsample(scale_factor=5),
            nn.Conv2d(256, 128, 5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128), 
            nn.Upsample(scale_factor=5),
            nn.Conv2d(128, 32, 5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32), 
            nn.Upsample(scale_factor=5),
            nn.Conv2d(32, 4, 5, padding='same'), #changed out from 3 to 4
            nn.Tanh(),
            nn.MaxPool2d(6), #this is just a check
        )

        self.changeSize = nn.AdaptiveAvgPool3d((4,226,226))
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.reshape_size[0], self.reshape_size[1])
        x = self.decoder(x)
        x = self.changeSize(x)
        return x

class VGDiscriminator(nn.Module):
    def __init__(self, latent_dim=128, input_size=(4, 226, 226)):
        super().__init__()
    
        self.discriminator = nn.Sequential(
            nn.Conv2d(4, 32, 5, padding='same'), #How many channels in?
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(32, 128, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(128), #is num_features the channels in?
            nn.Conv2d(128, 256, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.BatchNorm2d(256), #is num_features the channels in?
            nn.Conv2d(256, 256, 5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256), #is num_features the channels in?
        )

        self.flatten = nn.Flatten()

        # Calculate the size of the linear layer
        with torch.no_grad():
            x = self.discriminator(torch.zeros(input_size).unsqueeze(0))
            self.feature_size = x.shape
            self.linear_size = self.flatten(x).shape[1]
            print(self.feature_size, self.linear_size)

        self.fc = nn.Sequential(
            nn.Linear(self.linear_size, latent_dim*4), 
            nn.ReLU(),
            nn.Linear(latent_dim*4, 1), 
            nn.Sigmoid(),
        )

    def forward(self, original_input, dec_output, rand_choice):
        #concatenate input together
        x = torch.cat((original_input, dec_output, rand_choice), 0)

        x = self.discriminator(x)
        x = self.flatten(x)
        x = self.fc(x) # fc means fully connected

        return x #other version returned the third layer result, the final result, and a result in between?

class VAEGAN(nn.Module):
    def __init__(self, latent_dim=128, input_size=(4,226,226)):
        super().__init__()

        self.encoder = VGEncoder(latent_dim, input_size)
        self.decoder = VGDecoder(latent_dim) # , self.encoder.linear_size, self.encoder.feature_size[2:]
        self.discriminator = VGDiscriminator(latent_dim, input_size)

        self.latent_dim = latent_dim

    #    need to send both decoder output and x into disc for gan
        # self.discriminator = VGDiscriminator(latent_dim, input_size)
    
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out['z'])

        #randomly select here!! Once we know how!!
        rand_sel = Variable(torch.randn(len(x), self.latent_dim), requires_grad=True)
        rand_sel = self.decoder(rand_sel)
        discriminator_out = self.discriminator(x, decoder_out, rand_sel) #This seems wrong?

        return {
            'discriminator_out': discriminator_out,
            'decoder_out': decoder_out,
            'encoder_out': encoder_out
        }

#to test
if __name__ == "__main__":
    #Test encoder
    # encoder = VGEncoder()
    # print(encoder) #what does this print?
    # x = torch.randn(1, 4, 226, 226) #our previous dimensions were 64, 1, 5, 5

    # x = encoder(x)
    # print(x)
    # print(x['z'].shape) #torch.Size([1, 128])

    # #Test decoder
    # decoder = VGDecoder()
    # print(decoder)
    # x = torch.randn(1, 128) #batch size 1, 128 feature vectors
    # x = decoder(x)
    # print(x.shape) #torch.Size([1, 3, 1250, 833]) but needs to be [1, 4, 1025, 862]? What is the formula?
    # # REECE!! Our output here needs to be same size as input. WHYYYYY DOESN'T IT WORKK!!!??

    #Test Discriminator
    # discriminator = VGDiscriminator()
    # print(discriminator)
    # x = torch.randn(1, 4, 226, 226)

    # x = discriminator(x, x, x)
    # print(x)
    # print(x.shape)

    # Test VAE_GAN
    # Can't test until dimensions match
    vae_gan = VAEGAN()
    print(vae_gan)
    x = torch.randn(1, 4, 226, 226, dtype=torch.float32) #floats because that's how the spectrogram gets processed
    x = vae_gan(x)
    #print(x)
    print(x['discriminator_out'].shape) #I doubt this is right - I think I have dis in wrong place

    #After the above works, try to convert to spectrogram, then .wav
