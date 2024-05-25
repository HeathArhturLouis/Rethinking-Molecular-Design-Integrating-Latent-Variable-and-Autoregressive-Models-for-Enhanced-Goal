## main model is defined here
# TODO: Clean dead includes
import h5py
import numpy as np
import argparse
import random
import sys, os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from loss_vanilla_vae import cross_entropy_calc


sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser

from pytorch_initializer import weights_init

# TODO: PARAMETERIZE
GRU_LAYERS = 3
'''

VAE-MODEL DEFINITION
'''

class VanillaMolVAE(nn.Module):
    def __init__(self, 
                latent_dim, 
                beta,
                max_decode_steps, # Seq Len max
                eps_std,
                vocab_size,
                pnorm_means,
                pnorm_stds,
                property_names,
                decoder_embedding_dim,
                device = 'cpu',
                padding_token = 0,
                reparam=True,
                decoder_mod_type = 'gru'): # Default picked based on what works for SD-VAE


        super(VanillaMolVAE, self).__init__()
        self.pnorm_means = pnorm_means
        self.pnorm_stds = pnorm_stds
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_decode_steps = max_decode_steps
        self.decoder_mod_type = decoder_mod_type
        self.decoder_embedding_dim = decoder_embedding_dim
        self.property_names = property_names
        self.property_size = len(property_names)
        self.beta = beta

        self.eps_std = eps_std
        self.device = device
        self.reparam = reparam

        # Define Encoder, Decoder and loss function
        self.encoder = CNNEncoder(max_len=self.max_decode_steps,
                                latent_dim=self.latent_dim,
                                vocab_size=self.vocab_size,
                                embedding_dim=self.decoder_embedding_dim,
                                device=self.device).to(device)

        self.state_decoder = StateDecoder(max_len=self.max_decode_steps, 
                                        prop_dim=len(self.property_names),
                                        vocab_size = self.vocab_size, 
                                        latent_dim=self.latent_dim, 
                                        module_type=self.decoder_mod_type, 
                                        device=self.device).to(device)

        self.loss_fnct = cross_entropy_calc
        self.kl_coeff = beta

    def normalize_prop_scores(self, properties):
        '''
        Auxiliary function to be used by trainer/sampler class
        properties is torch.tensor of size num_scores x num_properties

        returns normalized properties
        '''
        assert properties.shape[-1] == self.property_size

        for i in range(self.property_size):
            # Apply ith index of self.pnorm_means and self.pnorm_stds to normalize every ith element of the 
            # num_properties dimension of properties
            properties[:, i] -= self.pnorm_means[i]
            properties[:, i] /= self.pnorm_stds[i]

        return properties

    def reparameterize(self, mu, logvar):
        if self.reparam:
            eps = mu.data.new(mu.size()).normal_(0, self.eps_std)
            
            if self.device == 'gpu':
                eps = eps.cuda()

            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, y_inputs, true_binary, return_logits = False):
        '''
        If return logits also returns raw logits
        '''
        # FWD pass through encoder
        z_mean, z_log_var = self.encoder(x_inputs)
        # Sampling
        
        z = self.reparameterize(z_mean, z_log_var)
        # Get raw logits 
        raw_logits = self.state_decoder(z, y_inputs)

        recon_loss = self.loss_fnct(true_tokens=true_binary.permute(1, 0), raw_logits=raw_logits)

        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        if not return_logits:
            return recon_loss, self.kl_coeff * torch.mean(kl_loss)
        else:
            return recon_loss, self.kl_coeff * torch.mean(kl_loss), raw_logits


    @property
    def config(self):
        return dict(latent_dim=self.latent_dim,
                    beta=self.beta,
                    max_decode_steps= self.max_decode_steps,
                    eps_std=self.eps_std,
                    vocab_size=self.vocab_size,
                    pnorm_means=list(self.pnorm_means),
                    pnorm_stds=list(self.pnorm_stds),
                    property_names=list(self.property_names),
                    decoder_embedding_dim=self.decoder_embedding_dim,
                    device=self.device,
                    # TODO: Make configurable
                    # padding_token=self.padding_token,
                    reparam=self.reparam,
                    decoder_mod_type=self.decoder_mod_type)


# encoder and decoder

class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim, vocab_size, embedding_dim, device):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # TODO: Parameterise convolution dimensions
        self.conv1 = nn.Conv1d(embedding_dim, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)
        self.mean_w = nn.Linear(435, self.latent_dim)
        self.log_var_w = nn.Linear(435, self.latent_dim)
        
        
        weights_init(self)

    def forward(self, x_cpu):
        if self.device == 'cpu':
            # LOUIS:
            # Variable() no longer neccesary in PyTorch > 0.4.0
            # I've removed the Variable() formerly enclosing torch.from_numpy(x_cpu)[.cuda()]
            # batch_input = torch.from_numpy(x_cpu)

            # Assume it arrives in a tensor
            batch_input = x_cpu.long()
        else:
            # Added .long to ensure tokens are ints
            batch_input = x_cpu.cuda().long()
            # batch_input = torch.from_numpy(x_cpu).cuda().long()

        embedded = self.embedding(batch_input)
        # Assuming the input is shape (batch_size, seq_length) and embedding output is (batch_size, seq_length, embedding_dim)
        # Conv1D expects (batch_size, channels, length), so we permute dimensions
        embedded_p = embedded.permute(0, 2, 1)

        # Permuted embedding is size batch_size x embedding_dimension x seqence_length
        #                               64                 47                 101
        h1 = self.conv1(embedded_p)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

         # Works when:
        # max_len is 100
        # embedding_dim is 80\
        # For me embedding_dim is 47
        # and max_len is 101

        # LOUIS t = 0: Tried commenting out jic
        #h3 = torch.transpose(h3, 1, 2).contiguous()

        # The use of view(x_cpu.shape[0], -1) here is suspicious; Try switching it to reshape
        flatten = h3.reshape(x_cpu.shape[0], -1)

        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        # LOUIS t = 1: The trace now points here, so it happens after z_log_car
        z_log_var = self.log_var_w(h)

        return (z_mean, z_log_var)    
    

class StateDecoder(nn.Module):
    def __init__(self, max_len, prop_dim, vocab_size, latent_dim, device, module_type = 'gru'):
        super(StateDecoder, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.module_type = module_type
        # TODO: Make adjustable
        # self.hidden_sz = latent_dim + cmd_args.output_dim
        self.n_props = prop_dim
        self.latent_dim = latent_dim + self.n_props

        self.max_len = max_len

        # From latent dim + prop size to latent dim
        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if self.module_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, GRU_LAYERS)
        # elif cmd_args.rnn_type == 'sru':
        #    self.gru = SRU(self.latent_dim, 501, 1)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, self.vocab_size)

        weights_init(self)

    def forward(self, z, y):
        if self.device == 'cpu':
            y = torch.tensor(y).float()
        else:
            y = torch.tensor(y).cuda().float() # .detach()

        assert len(z.size()) == 2 # assert the input is a matrix

        ztl_inputs = torch.cat((z, y.view(y.shape[0],1)), 1)

        h = self.z_to_latent(ztl_inputs)
        #h = self.z_to_latent(torch.cat((z, torch.tensor(y).cuda().float()), 1))

        h = F.relu(h)

        # Putput size is also vocabulary size, since it's one output per logit
        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1] + self.n_props) # repeat along time steps

        out, _ = self.gru(rep_h) # run multi-layer gru
        logits = self.decoded_logits(out)
        return logits   


    
