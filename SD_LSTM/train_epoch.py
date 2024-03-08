# import h5py
# import numpy as np
# import argparse
# import random
# import sys, os
# import torch
# from torch.autograd import Variable
# from torch.nn.parameter import Parameter
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Not neccesary?
# from cmd_args import cmd_args


def epoch_train_vanilla(phase, epoch, model, sample_idxes, data_binary, data_masks, data_property, cmd_args, optimizer):
    '''
    No regularizer/regressor, just vanilla VAE

    Input:
     - phase ('train' | 'valid') 
     - epoch (int : epoch number)
     - model (SD-LSTM model)
     - sample_idxes 
     - data_binary
     - data_masks
     - data_property
     - cmd_args (training parameters)
     - optimizer (optimizer for model)

    Returns:
        - model <-- the autoencoder model
        - avg_loss <-- average loss over all batches in the epoch
    
    Train model for 1 epoch
    '''

    # total_vae_loss = []  # perp loss total per batch
    # total_kl_loss = []  # KL loss total per batch
    total_loss = []

    pbar = tqdm(range(0, (len(sample_idxes) + (cmd_args.batch_size - 1)) // cmd_args.batch_size), unit='batch')

    if phase == 'train' and optimizer is not None:
        model.train()
    else:
        model.eval()

    n_samples = 0

    for pos in pbar:
        selected_idx = sample_idxes[pos * cmd_args.batch_size: (pos + 1) * cmd_args.batch_size]
        x_inputs, y_inputs, v_tb, v_ms, t_y = get_batch_input_vae(selected_idx, data_binary, data_masks, data_property)
        
        
        loss_list = model.forward(x_inputs, y_inputs, v_tb, v_ms, t_y)
        perp_loss = loss_list[0]
        kl_loss = loss_list[1]

        total_loss_batch = perp_loss + cmd_args.kl_coeff * kl_loss  # KL Divergence weighting coeff (\beta?)

        pbar.set_description(f'Epoch: {epoch} Phase: {phase} - VAE Loss: {total_loss_batch.item():.5f} | Perp Loss {perp_loss.item()} | KDL Loss {kl_loss.item()} |')

        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad()
            if optimizer_decoder is not None:
                optimizer_decoder.zero_grad()

            total_loss_batch.backward()

            optimizer_encoder.step()
            if optimizer_decoder is not None:
                optimizer_decoder.step()

        # Collect loss components
        total_vae_loss.append(perp_loss.item() * len(selected_idx))
        total_kl_loss.append(kl_loss.item() * len(selected_idx))
        n_samples += len(selected_idx)

    # Calculate the average loss of a batch
    avg_vae_loss = sum(total_vae_loss) / n_samples
    avg_kl_loss = sum(total_kl_loss) / n_samples 
    avg_loss = [avg_vae_loss, avg_kl_loss]

    return model, avg_loss



def train_batch(model, phase, in_samps, in_masks, in_prop, optimizer):
    '''
    Train model for on a single batch of data.
    
    Inputs: 
     - model <-- SD-LSTM instance
     - phase <-- either 'train' or 'valid', determines weather to step optimizer
     - in_samps <-- input grammar encodings [format: ]
     - in_masks <-- input grammar encoding masks [format: ]
     - in_prop <-- input properties
     - optimizer <-- optimizer [implements .step]

    Return:
     - Loss (TODO: figure out the format)

    Assumptions:
     - Everything is on correct device
     - Everything is correct dimnesions (batch_size)
     - Model is already in 'train' or 'valid' mode corresponding to phase argument
    '''
    if phase == 'train' and optimizer is not None:
        # Zero Grads
        optimizer.zero_grad()

    # Forward Pass
    raw_logits = self.state_decoder(z, y_inputs)
    # Loss Computation
    total_loss = model.perp_calc(true_binary, rule_masks, raw_logits)

    if phase == 'train' and optimizer is not None:
        # Backward Pass
        total_loss.backward()
        # Optimizer Step
        optimizer.step()



if __name__ == '__main__':
    pass
    # TODO: Write testing code etc...
    # model = 
    # model.device()
    # model.train()
    # Transform input data into next token selections
    #       Find where end of derivation is
    #       Select random token up to this point
    #       Does this mess up my code?