## main model is defined here
import numpy as np
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

sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser

#VAE model
## define the model
# from mol_util import rule_ranges, terminal_idxes, DECISION_DIM

from sd_loss_computation import PerpCalculator



class SDVAE(nn.Module):
    def __init__(self, 
                 latent_dim,
                 eps_std,
                 max_decode_steps,
                 beta,
                 device,
                 decision_dim,
                 decoder_rnn_type,
                 reparam,
                 pnorm_means,
                 pnorm_stds,
                 property_names,
                 loss_type='binary'):
        
        super(SDVAE, self).__init__()

        # Model Parameters
        self.latent_dim = latent_dim
        self.max_decode_steps = max_decode_steps
        self.beta = beta
        self.decoder_rnn_type = decoder_rnn_type
        self.device = device
        self.eps_std = eps_std
        self.reparam = reparam
        self.decision_dim = decision_dim
        self.property_names = property_names
        self.loss_type = loss_type

        self.prop_dim = len(self.property_names)


        self.pnorm_means=pnorm_means
        self.pnorm_stds=pnorm_stds


        self.encoder = CNNEncoder(max_len=self.max_decode_steps,
                                  latent_dim=self.latent_dim, 
                                  device=self.device, 
                                  decision_dim=self.decision_dim).to(device)

        self.state_decoder = StateDecoder(max_len=self.max_decode_steps,
                                         latent_dim=self.latent_dim,
                                         prop_dim = self.prop_dim,
                                         rnn_type = self.decoder_rnn_type,
                                         decision_dim = self.decision_dim,
                                         device=self.device).to(device)
        
        # Loss calculator
        self.perp_calc = PerpCalculator(loss_type=self.loss_type)
        

    def reparameterize(self, mu, logvar):
        '''LOUIS:
        Reparameterization Trick Refresher:
            - Encoders output are [\mu, \sigma]  q(z|x) = N(z; \mu, \sigma^2)
            - Sampling directly from N(z;\mu, \sigma^2) can't be backrpoped over
            - Instead, sample \epsilon \sim N(0, I) [independent of params]
            - Reparameterize z as z = \mu + \sigma \times \epsilon
            - This keeps the probabilistic nature of z, but it's relationship with \mu \sigma is deterministic so we can compute gradients
        '''

        # RT only needs to be applied during training to allow for backprop. During inference we can ignore it
        if self.reparam:
            # Sample epsilon from desired distribution (by default cmd_args.eps_std = 0.01) TODO: Why do we want less variability?
            eps = mu.data.new(mu.size()).normal_(0, self.eps_std)
            
            # Send to GPU if required
            if self.device == 'cuda':
                eps = eps.cuda()
            
            # LOUIS: This is no longer neccesary
            # eps = Variable(eps)
            
            # Reparameterization (logvar to std dev is exp(1/2*\sig^2))
            return mu + eps * torch.exp(logvar * 0.5)            
        else:
            return mu

    def forward(self, x_inputs, y_inputs, true_binary, rule_masks):
        # FWD pass through encoder
        z_mean, z_log_var = self.encoder(x_inputs)
        # Sampling
        z = self.reparameterize(z_mean, z_log_var)
        # Get raw logits 
        raw_logits = self.state_decoder(z, y_inputs)

        # Raw logits are seq_len x batch_size x decision dim

        # Other two batch_first

        perplexity = self.perp_calc(true_binary, rule_masks, raw_logits)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), -1)
        
        return perplexity, self.beta * torch.mean(kl_loss)


    def normalize_prop_scores(self, properties):
        '''
        returns properties normalized by model normalization parameters 
        (should be used for both training and inference)
        '''
        assert properties.shape[-1] == self.prop_dim

        for i in range(self.prop_dim):
            # Apply ith index of self.pnorm_means and self.pnorm_stds to normalize every ith element of the 
            # num_properties dimension of properties
            properties[:, i] -= self.pnorm_means[i]
            properties[:, i] /= self.pnorm_stds[i]
        
        return properties

    @property
    def config(self):
        return dict(latent_dim=self.latent_dim,
                    decision_dim=self.decision_dim,
                    eps_std = self.eps_std,
                    max_decode_steps = self.max_decode_steps,
                    beta = self.beta,
                    decoder_rnn_type = self.decoder_rnn_type,
                    device = self.device,
                    reparam = self.reparam,
                    pnorm_means=list(self.pnorm_means),
                    pnorm_stds=list(self.pnorm_stds),
                    property_names=list(self.property_names),
                    loss_type=self.loss_type,
                    )


# encoder and decoder
from pytorch_initializer import weights_init
#q(z|x)
class CNNEncoder(nn.Module):
    def __init__(self, max_len, latent_dim, device, decision_dim):
        super(CNNEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.decision_dim = decision_dim

        self.max_len = max_len

        self.conv1 = nn.Conv1d(self.decision_dim, 9, 9)
        self.conv2 = nn.Conv1d(9, 9, 9)
        self.conv3 = nn.Conv1d(9, 10, 11)

        self.last_conv_size = self.max_len - 9 + 1 - 9 + 1 - 11 + 1
        self.w1 = nn.Linear(self.last_conv_size * 10, 435)

        self.mean_w = nn.Linear(435, self.latent_dim)
        self.log_var_w = nn.Linear(435, self.latent_dim)
        weights_init(self)

    def forward(self, x_cpu):
        if self.device == 'cpu':
            # batch_input = torch.from_numpy(x_cpu)
            batch_input = x_cpu
        else:
            # batch_input = torch.from_numpy(x_cpu).cuda()
            batch_input = x_cpu.cuda()

        h1 = self.conv1(batch_input)
        h1 = F.relu(h1)
        h2 = self.conv2(h1)
        h2 = F.relu(h2)
        h3 = self.conv3(h2)
        h3 = F.relu(h3)

        # The use of view(x_cpu.shape[0], -1) here is suspicious; Try switching it to reshape 
        flatten = h3.reshape(x_cpu.shape[0], -1)
        # Flatten is shape batch_size x 730 <-- should be 740?
        
        h = self.w1(flatten)
        h = F.relu(h)

        z_mean = self.mean_w(h)
        # LOUIS t = 1: The trace now points here, so it happens after z_log_car
        z_log_var = self.log_var_w(h)

        return (z_mean, z_log_var)    
    

class StateDecoder(nn.Module):
    def __init__(self, max_len, latent_dim, prop_dim, rnn_type, decision_dim, device):
        super(StateDecoder, self).__init__()
        # self.output_dim = output_dim

        self.prop_dim = prop_dim
        self.latent_dim = latent_dim + self.prop_dim
        self.max_len = max_len
        self.decision_dim = decision_dim
        self.device = device

        self.z_to_latent = nn.Linear(self.latent_dim, self.latent_dim)
        if rnn_type == 'gru':
            self.gru = nn.GRU(self.latent_dim, 501, 1)
        elif rnn_type == 'sru':
            self.gru = SRU(self.latent_dim, 501, 1)
        else:
            raise NotImplementedError

        self.decoded_logits = nn.Linear(501, self.decision_dim)
        weights_init(self)

    def forward(self, z, y):
        if self.device == 'cpu':
            y = torch.tensor(y).float()
        else:
            y = torch.tensor(y).cuda().float() # .detach()

        assert len(z.size()) == 2 # assert the input is a matrix
        h = self.z_to_latent(torch.cat((z, y.view(y.shape[0],1)), 1))
        #h = self.z_to_latent(torch.cat((z, torch.tensor(y).cuda().float()), 1))

        h = F.relu(h)
        rep_h = h.expand(self.max_len, z.size()[0], z.size()[1] + self.prop_dim) # repeat along time steps
        out, _ = self.gru(rep_h) # run multi-layer gru
        logits = self.decoded_logits(out)
        return logits   
