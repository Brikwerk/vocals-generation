import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80+dim_emb if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x = x.squeeze(1).transpose(2,1)
        # c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        # x = torch.cat((x, c_org), dim=1)
        
        x = x.transpose(2, 1)
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        out_forward = outputs[:, :, :self.dim_neck]
        out_backward = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, outputs.size(1), self.freq):
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes
      
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, dim_neck_decoder=False):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        if dim_neck_decoder:
            self.decoder = Decoder(dim_neck_decoder, dim_emb, dim_pre)
        else:
            self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, return_intermediate=False, return_encoder_output=False,
                condition_vec=None):
                
        codes = self.encoder(x)
        if return_intermediate:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        encoder_outputs = torch.cat(tmp, dim=1)

        if return_encoder_output:
            return encoder_outputs

        if condition_vec is not None:
            encoder_outputs = torch.cat((encoder_outputs, condition_vec), dim=-1)
        
        # encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)


class GeneratorV2(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq, seq_length, bottleneck_dim,
                 dim_neck_decoder=False):
        super(GeneratorV2, self).__init__()

        self.seq_length = seq_length
        self.dim_neck = dim_neck
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        if dim_neck_decoder:
            self.decoder = Decoder(dim_neck_decoder, dim_emb, dim_pre)
        else:
            self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

        self.cond_proj = nn.Linear((dim_neck*2) * seq_length, bottleneck_dim)
        self.vocals_proj = nn.Linear((dim_neck*2) * seq_length, bottleneck_dim)
        self.latent_proj = nn.Linear(bottleneck_dim*2, (dim_neck*2) * seq_length)

        self.mu_fc = nn.Linear(bottleneck_dim*2, bottleneck_dim*2)
        self.logvar_fc = nn.Linear(bottleneck_dim*2, bottleneck_dim*2)
        self.logscale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, x, return_intermediate=False, return_encoder_output=False,
                condition_vec=None):
                
        codes = self.encoder(x)
        if return_intermediate:
            return torch.cat(codes, dim=-1)
        
        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        encoder_outputs = torch.cat(tmp, dim=1)

        if return_encoder_output:
            return encoder_outputs

        z = None
        mu = None
        std = None
        if condition_vec is not None:
            # Project condition and vocals tensors into bottleneck space
            condition_vec = condition_vec.flatten(start_dim=1)
            condition_vec = self.cond_proj(condition_vec)
            vocals_vec = self.vocals_proj(encoder_outputs.flatten(start_dim=1))

            encoder_outputs = torch.cat((vocals_vec, condition_vec), dim=-1)

            # Reparameterization trick
            mu = self.mu_fc(encoder_outputs)
            logvar = self.logvar_fc(encoder_outputs)
            std = torch.exp(logvar / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()

            encoder_outputs = self.latent_proj(z)
            encoder_outputs = encoder_outputs.reshape(z.size(0), self.seq_length, -1)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1), z, mu, std


if __name__ == "__main__":
    g = GeneratorV2(160, 0, 512, 20, 860, 128)
    g(torch.randn(1, 860, 80), torch.randn(1, 860, 320))
    
