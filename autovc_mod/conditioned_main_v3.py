import os
import argparse
from random import random
from solver_encoder import ConditionedSolverV2
from data_loader import get_loader_combined
import torch
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Reproducibility
    torch.manual_seed(42)

    # Data loader.
    vcc_loader, val_loader = get_loader_combined(
        config.data_dir, config.batch_size, config.len_crop, config.random_flip,
        config.random_crop)
    
    solver = ConditionedSolverV2(vcc_loader, val_loader, config)

    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=160)
    parser.add_argument('--dim_emb', type=int, default=0)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=20)
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    
    # Training configuration.
    parser.add_argument('--c_weights_dir', type=str, default='model_latest_accom.pth')
    parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=860, help='dataloader output sequence length')
    parser.add_argument('--random_flip', type=str2bool, default=False, help='whether to randomly flip the audio')
    parser.add_argument('--random_crop', type=str2bool, default=False, help='whether to randomly crop the audio')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=250)

    config = parser.parse_args()
    print(config)
    main(config)