import os
import torch
import torch.nn.functional as F
import time
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from losses import gaussian_likelihood, kl_divergence
from model_vc import Generator, GeneratorV2


class Solver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model()

            
    def build_model(self):
        
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt']

        lowest_loss = None
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real = next(data_iter)
            
            
            x_real = x_real.to(self.device) 
            # emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real)

            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)

            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, return_intermediate=True)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            # loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

                # Plot original and reconstructed spectrograms
                fig = plt.figure(figsize=(10, 15), dpi=300)
                ax = fig.add_subplot(211)
                ax.imshow(x_real[0].cpu().detach().numpy(), aspect='auto', origin='lower')
                ax.set_title('Original')
                ax2 = fig.add_subplot(212)
                ax2.imshow(x_identic_psnt[0].cpu().detach().numpy(), aspect='auto', origin='lower')
                ax2.set_title('Reconstructed')
                plt.savefig(os.path.join('output.png'))
                plt.close()

                torch.save(self.G.state_dict(), os.path.join(f'model_latest.pth'))

            if lowest_loss is None or g_loss.item() < lowest_loss:
                print('New lowest loss, saving model.')
                lowest_loss = g_loss_id.item()
                torch.save(self.G.state_dict(), os.path.join(f'model.pth'))
                

class ConditionedSolver(object):

    def __init__(self, vcc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model(config.c_weights_dir)

            
    def build_model(self, condition_weights_path):

        self.conditioned_G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.conditioned_G.load_state_dict(torch.load(condition_weights_path))
        # Freeze conditioned generator
        for param in self.conditioned_G.parameters():
            param.requires_grad = False
        
        # Dim neck for the decoder is 2 times the size to account for
        # both the conditioned latent space and the encoder's latent space.
        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.conditioned_G.to(self.device)
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
    
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt']

        writer = SummaryWriter()
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real_accom, x_real_vocals = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_accom, x_real_vocals = next(data_iter)
            
            
            x_real_accom, x_real_vocals = x_real_accom.to(self.device), x_real_vocals.to(self.device)
            # emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            # Get the latent space vector from the conditioned generator
            with torch.no_grad():
                accom_encoded = self.conditioned_G(x_real_accom, return_encoder_output=True)
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real = self.G(x_real_vocals, condition_vec=accom_encoded)

            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)

            g_loss_id = F.mse_loss(x_real_vocals, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real_vocals, x_identic_psnt)
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, return_intermediate=True)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            # loss['G/loss_cd'] = g_loss_cd.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            writer.add_scalar("Loss/train", g_loss.item(), i)

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

                original = x_real_vocals[0].unsqueeze(0).cpu()
                reconstructed = x_identic_psnt[0].unsqueeze(0).cpu()

                # Normalize to [0,1]
                original = (original - original.min()) / (original.max() - original.min())
                reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

                writer.add_image('Original', original, i)
                writer.add_image('Reconstructed', reconstructed, i)

                torch.save(self.G.state_dict(), os.path.join(f'model_latest.pth'))

        writer.flush()
        writer.close()


class ConditionedSolverV2(object):

    def __init__(self, vcc_loader, val_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.vcc_loader = vcc_loader
        self.val_loader = val_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.seq_length = config.len_crop
        self.bottleneck_dim = config.bottleneck_dim

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step

        # Build the model and tensorboard.
        self.build_model(config.c_weights_dir)

            
    def build_model(self, condition_weights_path):

        self.conditioned_G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)
        self.conditioned_G.load_state_dict(torch.load(condition_weights_path))
        # Freeze conditioned generator
        for param in self.conditioned_G.parameters():
            param.requires_grad = False
        
        # Dim neck for the decoder is 2 times the size to account for
        # both the conditioned latent space and the encoder's latent space.
        self.G = GeneratorV2(self.dim_neck, self.dim_emb, self.dim_pre,
                             self.freq, self.seq_length, self.bottleneck_dim)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)
        
        self.conditioned_G.to(self.device)
        self.G.to(self.device)
        

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#


    def validation(self):
        """Validate the model on the validation set."""
        self.G.eval()

        losses = []
        with torch.no_grad():
            for i, (x_real_accom, x_real_vocals) in enumerate(self.val_loader):
                x_real_accom, x_real_vocals = x_real_accom.to(self.device), x_real_vocals.to(self.device)

                accom_encoded = self.conditioned_G(x_real_accom, return_encoder_output=True)

                # Identity mapping loss
                x_identic, x_identic_psnt, code_real, z, mu, std = self.G(x_real_vocals, condition_vec=accom_encoded)

                x_identic = x_identic.squeeze(1)
                x_identic_psnt = x_identic_psnt.squeeze(1)

                # g_loss_id = F.mse_loss(x_real_vocals, x_identic)   
                # g_loss_id_psnt = F.mse_loss(x_real_vocals, x_identic_psnt)

                kl_loss = kl_divergence(z, mu, std)

                base_recon_loss = gaussian_likelihood(x_identic.unsqueeze(1), self.G.logscale, x_real_vocals.unsqueeze(1))
                base_recon_loss = kl_loss - base_recon_loss
                base_recon_loss = base_recon_loss.mean()
                postnet_recon_loss = gaussian_likelihood(x_identic_psnt.unsqueeze(1), self.G.logscale, x_real_vocals.unsqueeze(1))
                postnet_recon_loss = kl_loss - postnet_recon_loss
                postnet_recon_loss = postnet_recon_loss.mean()
                
                # Code semantic loss.
                code_reconst = self.G(x_identic_psnt, return_intermediate=True)
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                # Backward and optimize.
                # g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
                g_loss = base_recon_loss + postnet_recon_loss + self.lambda_cd * g_loss_cd

                losses.append(g_loss.item())

        self.G.train()

        return np.mean(losses)
    
                
    def train(self):
        # Set data loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt']

        writer = SummaryWriter()
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        lowest_val_loss = None
        for i in range(self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            try:
                x_real_accom, x_real_vocals = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real_accom, x_real_vocals = next(data_iter)
            
            
            x_real_accom, x_real_vocals = x_real_accom.to(self.device), x_real_vocals.to(self.device)
            # emb_org = emb_org.to(self.device) 
                        
       
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            # Get the latent space vector from the conditioned generator
            with torch.no_grad():
                accom_encoded = self.conditioned_G(x_real_accom, return_encoder_output=True)
            
            self.G = self.G.train()
                        
            # Identity mapping loss
            x_identic, x_identic_psnt, code_real, z, mu, std = self.G(x_real_vocals, condition_vec=accom_encoded)

            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)

            # g_loss_id = F.mse_loss(x_real_vocals, x_identic)   
            # g_loss_id_psnt = F.mse_loss(x_real_vocals, x_identic_psnt)

            kl_loss = kl_divergence(z, mu, std)

            base_recon_loss = gaussian_likelihood(x_identic.unsqueeze(1), self.G.logscale, x_real_vocals.unsqueeze(1))
            base_recon_loss = kl_loss - base_recon_loss
            base_recon_loss = base_recon_loss.mean()
            postnet_recon_loss = gaussian_likelihood(x_identic_psnt.unsqueeze(1), self.G.logscale, x_real_vocals.unsqueeze(1))
            postnet_recon_loss = kl_loss - postnet_recon_loss
            postnet_recon_loss = postnet_recon_loss.mean()
            
            # Code semantic loss.
            code_reconst = self.G(x_identic_psnt, return_intermediate=True)
            g_loss_cd = F.l1_loss(code_real, code_reconst)


            # Backward and optimize.
            # g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
            g_loss = base_recon_loss + postnet_recon_loss + self.lambda_cd * g_loss_cd
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            # loss['G/loss_id'] = g_loss_id.item()
            # loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            # loss['G/loss_cd'] = g_loss_cd.item()

            loss['G/loss_id'] = base_recon_loss.item()
            loss['G/loss_id_psnt'] = postnet_recon_loss.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            writer.add_scalar("Loss/train", g_loss.item(), i)

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)

                # Perform validation
                val_loss = self.validation()
                print('\tValidation loss:', val_loss)
                writer.add_scalar("Loss/val", val_loss, i)
                if lowest_val_loss is None or val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    print('\tAchieved lowest val loss, saving model...')
                    torch.save(self.G.state_dict(), os.path.join(f'model_lowest_val_vae.pth'))

                original = x_real_vocals[0].unsqueeze(0).cpu()
                reconstructed = x_identic_psnt[0].unsqueeze(0).cpu()

                # Normalize to [0,1]
                original = (original - original.min()) / (original.max() - original.min())
                reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

                writer.add_image('Original', original, i)
                writer.add_image('Reconstructed', reconstructed, i)

                torch.save(self.G.state_dict(), os.path.join(f'model_latest_vae.pth'))

        writer.flush()
        writer.close()
