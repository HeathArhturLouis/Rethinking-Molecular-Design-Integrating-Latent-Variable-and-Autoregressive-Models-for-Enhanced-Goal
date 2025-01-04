
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import sys

import os

import numpy as np
from tqdm import tqdm

sys.path.append('../utils/')
from rnn_utils import load_smiles_and_properties, get_tensor_dataset, save_model, load_model, rnn_start_token_vector

sys.path.append('../LSTM/')
from reg_rnn_model import ConditionalSmilesRnn


from smiles_char_dict import SmilesCharDictionary


from haikunator import Haikunator
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


from model_vanilla_vae import VanillaMolVAE

from torch.distributions import Categorical

import math

import wandb

N_WORKERS_DL = 4


class PolicyRegularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs_decoder, log_probs_lstm):
        ctx.save_for_backward(log_probs_lstm, log_probs_decoder)

        return (-1 * log_probs_lstm.mean()) # The mean of this when returned will be our final loss

    @staticmethod
    def backward(ctx, grad_output):
        # Grad output contains grad of 

        log_probs_lstm, log_probs_decoder = ctx.saved_tensors #, log_probs_decoder = ctx.saved_tensors

        weighted_grads = grad_output * log_probs_decoder # grad_output * log_probs_lstm * log_probs_decoder

        # Return None for x_inputs and gradients for decoder parameters
        return weighted_grads, None

class VanillaVAETrainerReg:
    def __init__(self,
                 regularizer_type, # in [None, 'KLD', 'Pol']
                 regularizer_model_path, # Path to LSTM Model
                 regularizer_param_path,
                 regularizer_weight, # Coefficient weighting regularizer loss 
                 train_x_enc, train_y, valid_x_enc, valid_y, property_names,
                 model, device, batch_size,
                 latent_dim=56,
                 beta=1.0,
                 max_seq_len=100,
                 eps_std=0.01,
                 decoder_embedding_dim=47,# Prev was tok size
                 learning_rate = 1e-3,  model_save_dir = '../models/VANILLA_VAE_REGULARIZED/', 
                 save_every_n_val_cycles = 3, max_epochs = 500, wandb_project = None, early_stopping_epochs = 15,
                 preg_latent_sample_size=4,# Regularizer specific parameters
                 preg_decoder_sample_size=4,
                 n_lstm_samples = 10,
                 prior_std = 1.0,
                 beta_annealing_function=None,
                 reg_weight_annealing_function=None,
                 decoder_mod_type = 'rnn'):

        #### Annealing Function ####
        if beta_annealing_function is None:
            self.beta_annealing_function = (lambda beta, epoch : beta)
        else:
            self.beta_annealing_function = beta_annealing_function

        if reg_weight_annealing_function is None:
            self.reg_weight_annealing_function = (lambda reg_weight, epoch: reg_weight)
        else:
            self.reg_weight_annealing_function = reg_weight_annealing_function


        assert regularizer_type in [None, 'KLD', 'Pol', 'Pol2']

        self.unconditional = False
        self.n_lstm_samples = n_lstm_samples
        self.regularizer_type = regularizer_type
        self.early_stopping_epochs = early_stopping_epochs
        self.prior_std = prior_std

        if self.regularizer_type is not None:
            self.reg_model = load_model(model_class=ConditionalSmilesRnn, 
                                        model_definition=regularizer_param_path, 
                                        model_weights=regularizer_model_path, 
                                        device=device)
            
            self.reg_model = self.reg_model.eval()
            self.regularizer_weight = regularizer_weight
            self.regularizer_model_path = regularizer_model_path
        else:
            self.reg_model = None
            self.regularizer_model_path = 'No regularizer used'
            self.regularizer_weight = 0


        if self.regularizer_type != 'Pol':
            self.preg_latent_sample_size= 0
            self.preg_decoder_sample_size= 0
        
        if self.regularizer_type != 'Pol2':
            self.n_lstm_samples = 0
        
        if self.regularizer_type == 'Pol':
            self.preg_latent_sample_size = preg_latent_sample_size
            self.preg_decoder_sample_size = preg_decoder_sample_size
        if self.regularizer_type =='pol2':
            self.n_lstm_samples = n_lstm_samples

        self.distribution_cls = Categorical

        # Normalize y_properties
        all_y = np.concatenate((train_y, valid_y), axis=0)
        self.pnorm_means = np.mean(all_y, axis = 0)
        self.pnorm_stds = np.std(all_y, axis = 0)
        self.n_props = train_y.shape[-1] # Should be 1

        # Model Save Params
        self.property_names = property_names
        self.max_seq_len = max_seq_len
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.char_dict = SmilesCharDictionary()
        self.vocab_size = self.char_dict.get_char_num()
        self.eps_std = eps_std
        self.beta = beta
        
        if model is None:
            self.model = VanillaMolVAE(
                            property_names=self.property_names,
                            latent_dim=latent_dim,
                            beta=self.beta,
                            max_decode_steps = self.max_seq_len, # Seq Len max
                            eps_std = self.eps_std,
                            vocab_size = self.vocab_size,
                            pnorm_means = self.pnorm_means,
                            pnorm_stds = self.pnorm_stds,
                            device = self.device,
                            decoder_embedding_dim=decoder_embedding_dim,
                            padding_token = 0,
                            reparam=True,
                            decoder_mod_type = decoder_mod_type,
                            )
                
        else:
            self.model = model.to(device)

        # Normalize property scores
        train_y = self.model.normalize_prop_scores(train_y)
        valid_y = self.model.normalize_prop_scores(valid_y)
        
        train_set = get_tensor_dataset(train_x_enc, train_y)
        valid_set = get_tensor_dataset(valid_x_enc, valid_y)

        self.train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS_DL)
        self.val_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS_DL)

        self.save_every_n_val_cycles = save_every_n_val_cycles

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.reg_optimizer = torch.optim.Adam(self.model.state_decoder.parameters())

        self.max_epochs = max_epochs

        self.num_reg_samples = 1

        self.train_losses = []
        self.val_losses = []

        self.model_save_dir = model_save_dir

        if batch_size is None:
            self.batch_size = train_data_loader.batch_size
        else:
            self.batch_size = batch_size

        os.makedirs(self.model_save_dir, exist_ok=True)

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, min_lr=0.00001)



        # Wandb tracking
        self.wandb_project = wandb_project

        if self.wandb_project != None:
            wandb.init(project=self.wandb_project)
            wandb.config = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'property_names':list(self.property_names),
            'property_normalizations_means': list(self.pnorm_means),
            'property_normalization_stds':list(self.pnorm_stds),
            'device':self.device,
            'eps_std':self.eps_std,
            'beta':self.beta,
            'vocab_size': self.vocab_size,  # Assuming input_size is accessible in your model
            'max_seq_len': self.max_seq_len,  # Assuming max_rules corresponds to max sequence length in your model
            'save_every_n_val_cycles': self.save_every_n_val_cycles,
            'regularizer_model_path' : self.regularizer_model_path,
            'regularizer_weight' : self.regularizer_weight,
            'early_stopping_after_epochs' : self.early_stopping_epochs,
            'preg_decoder_sample_size':self.preg_decoder_sample_size,
            'preg_latent_sample_size':self.preg_latent_sample_size,
            'n_lstm_samples':self.n_lstm_samples,
            'regularizer_prior_std':self.prior_std
            }

            # Watch model to track gradients
            # Probably inefficient
            # wandb.watch(self.model)
        
        self.sd = SmilesCharDictionary()

    def _sample_decoder_actions_and_likelihoods(self, input_latent, input_props, decoder_sample_size):
        '''
        Input:
        latent + property pairs, tensors should have same first dim

        - Sample decoder logits for input_latent latent points and input_props conditioning information.
        - Sample actions based on these and return the action / property pairs
        - For each latent prop pair sample decoder_sample_size different molecules from the decoder

        Do not sample around input_latent as they are already sampled from p(z)
        return:
        - Actions: (no_pairs * decoder_sample_size) x max_seq_len sequence_length sequence of actions for each new molecule
        - Log Probs: [(no_pairs * decoder_sample_size)] likelyhood of each molecule (based on selected actions)
        '''

        # TODO: Remove
        assert input_latent.shape[0] == input_props.shape[0]

        # Sample logits for each latent property pair
        decoder_logits = self.model.state_decoder(z=input_latent, y=input_props).permute([1, 0, 2])

        # remove?
        # output_actions = torch.zeros( [ decoder_logits.shape[0] * decoder_sample_size , self.model.max_decode_steps] , dtype=torch.int64, device = self.device)
        # output_likelyhoods = torch.zeros([ decoder_logits.shape[0] * decoder_sample_size] , dtype=torch.float32, device = self.device)

        # decoder_logits is : n_pairs x max_seq_len x decision_dim | 101, 8, 47
        # output_likelyhoods is : (n_pairs * decoder_sample_size) | 32
        # n_pairs is : 8
        # output_actions is : (n_pairs * decoder_sample_size) x max_seq_len | 32 x 101

        logits_expanded = decoder_logits.repeat_interleave(decoder_sample_size, dim=0)
        expanded_properties = input_props.repeat_interleave(decoder_sample_size, dim=0)

        '''
        logits_expanded = torch.tensor([
            [
                [0.01, 100, 0.01],
                [100, 0.01, 0.01],
                [0.01, 0.01, 100]
            ],
            [
                [100, 0.01, 0.01],
                [0.01, 0.01, 100],
                [0.01, 100, 0.01]
            ]
        ])
        '''
        logits_stable = logits_expanded - logits_expanded.max(dim=-1, keepdim=True)[0]
        probabilities = torch.softmax(logits_stable, dim= -1)

        probabilities_reshaped = probabilities.view(-1, probabilities.shape[-1])
        sampled_actions_flat = torch.multinomial(probabilities_reshaped, num_samples=1, replacement=True)
        sampled_actions = sampled_actions_flat.squeeze(-1).view(probabilities.shape[0], probabilities.shape[1])

        # compute probabilities based on the sampled actions
        log_probs = self.log_probs_from_logits(actions= sampled_actions, logits= logits_expanded)

        return log_probs, sampled_actions, expanded_properties

    def compute_log_probs(self, actions, logits, stability=True):
        if stability:
            min_log_value = -1e9
        
        log_probabilities = F.log_softmax(logits, dim=-1)
        log_probabilities = torch.clamp(log_probabilities, min=min_log_value)

        selected_log_probs = torch.gather(log_probabilities, 2, actions.unsqueeze(-1)).squeeze(-1)
        mask = (actions == self.sd.char_idx[self.sd.END])

        first_end_indices = mask.float().cumsum(dim=1).argmax(dim=1)

        no_end_mask = mask.sum(dim=1) == 0
        first_end_indices[no_end_mask] = actions.shape[1]
        
        # Change 1.
        valid_mask = torch.arange(actions.shape[1], device=self.device)[None, :] <= first_end_indices[:, None]
        # valid_mask[no_end_mask, 0] = True

        # Change 2.
        valid_mask[no_end_mask, :] = False
        
        # By default, if we want untermed to not be considered, we should set mask to False, then it will fill all logprobs with 0.0

        # print(valid_mask[0])
        # print(actions[0])
        # raise Exception

        masked_selected_log_probs = selected_log_probs.masked_fill(mask=(~valid_mask), value=0.0)
        sample_log_probs = torch.sum(masked_selected_log_probs, dim=1)

        return sample_log_probs

    def logits_fill(self, actions, logits):
        '''
        actions is what index is chosen and shape b_size x seq_len
        logits is logits and shape b_size x seq_len x decision_dim

        This function changes all logits after the first occurance of end token in action to [ 0.0001 .... Lots .... 0.0001] where lots
        is a large value at the index of decision_dim corresponding to END

        by doing this we're making it as if the model was trained to allways select the END token after asequence terminates instead of being trained
        to ignore these indecies (not computed for loss) and generating whatever 
        '''
        # actions[0, 3] = 2
        # actions[0, 9] = 2

        b_size, seq_len, decision_dim = logits.shape
        end_token = self.sd.char_idx[self.sd.END]

        mask = (actions == end_token)

        first_end_indices = torch.where(mask.any(dim=1), mask.float().argmax(dim=1), seq_len)

        if (first_end_indices == seq_len).all():
            return logits

        range_indices = torch.arange(seq_len, device=actions.device).expand(b_size, seq_len)

        valid_mask = range_indices > first_end_indices[:, None]

        valid_mask_expanded = valid_mask.unsqueeze(-1).expand(b_size, seq_len, decision_dim)

        fill_logits = torch.full((b_size, seq_len, decision_dim), 0.0001, device=logits.device)
        fill_logits[:, :, end_token] = 1e6  # Set a large value for the END token index

        # Apply the fill logits to positions after the first end character
        logits[valid_mask_expanded] = fill_logits[valid_mask_expanded]

        # print(actions[0])
        # print(logits[0, 0, :])
        # print(logits[0, 1, :])
        # print(logits[0, 2, :])
        # print(logits[0, 3, :])
        # print(logits[0, 4, :])
        # print(logits[0, 5, :])
        # raise Exception

        return logits



    def alternate_policy_regularizer(self, props, latent_sample_size = 4, decoder_sample_size = 10):
        '''
        props --> props
        latent_sample_size --> no latent points per prop
        decoder_sample_size --> no_decoder samples for each latent prop pair | should be large
        '''
        props_exp = props.repeat_interleave(latent_sample_size, dim=0)
        latent_points = torch.randn(props_exp.shape[0], self.model.latent_dim, device=self.model.device, dtype=torch.float32)  * self.prior_std # Should be 1

        '''
        props_exp --> 64 x 1
        latent_points --> 64, 56
        '''
        
        # Sample points and logits from the decoder
        decoder_logits = self.model.state_decoder(z=latent_points, y=props_exp).transpose(1, 0)

        '''
        Decoder logits is 64 x 37 x 47
        Decoder logits does not contain start token probs
        '''

        # Sample actions from decoder logits
        decoder_logits_exp = decoder_logits.repeat_interleave(decoder_sample_size, dim=0)
        '''
        decoder_logits_exp is 64 x 37 x 47 
        '''

        decoder_log_probs_exp = F.log_softmax(decoder_logits_exp, dim=-1)
        decoder_probs_exp = torch.exp(decoder_log_probs_exp)
        probabilities_flat = decoder_probs_exp.view(-1, decoder_logits_exp.shape[-1])
        actions_flat = torch.multinomial(probabilities_flat, num_samples=1)
        decoder_actions = actions_flat.view(decoder_logits_exp.shape[0], decoder_logits_exp.shape[1])
        
        # Compute decoder log probs
        decoder_log_probs = self.compute_log_probs(decoder_actions, decoder_logits_exp)

        # Get LSTM log probs
        with torch.no_grad():
            props_exp_exp = props_exp.repeat_interleave(decoder_sample_size, dim=0)

            start_tokens = torch.ones(
                                (props_exp_exp.shape[0], 1),
                                dtype=torch.int,
                                device = self.model.device
                                )

            # Expects input actions 
            input_actions = torch.cat([start_tokens, decoder_actions], dim=-1)

            hidden = self.reg_model.init_hidden(props_exp_exp.shape[0], self.model.device)

            lstm_actions, lstm_logits = self.reg_model.forward(
                                                x = input_actions,
                                                properties = props_exp_exp,
                                                hidden = hidden,
                                                use_teacher_forcing = True,
                                                sampling = True, # As opposed to argmax
                                                return_actions = False,
                                                return_both = True,
                                                seq_len = self.model.max_decode_steps + 1
                                            )


        # Change 3. LSTM log probs fill
        # lstm_logits = self.logits_fill(actions = lstm_actions, logits = lstm_logits)

        lstm_log_probs = self.compute_log_probs(actions = decoder_actions, logits = lstm_logits)

        # print('Finished')
        # raise Exception



        loss = -1 * (lstm_log_probs * decoder_log_probs).mean() * self.regularizer_weight + 1.6
        
        # Original Version:
        # return (lstm_log_probs * decoder_log_probs).mean() * -1 * self.regularizer_weight + 1.0
        
        # Module Versions:
        # loss = PolicyRegularizer.apply(decoder_log_probs, lstm_log_probs).mean() * self.regularizer_weight
        return loss



    def compute_policy_regularizer(self, props, prop_samp_size = None, latent_sample_size = 4, decoder_sample_size = 10, old_version= False):
        '''
        props <-- batch_size number of properties
        prop_sample_size <-- number of properties to sample
        latent_sample_size <-- number of latent points to sample per property
        decoder_sample_size <-- number of x to sample for decoder for each latent point
        '''

        '''
        1. Sample latent points
        '''

        if prop_samp_size is not None:
            raise Exception('Not yet implemented...')
            props = props[:prop_samp_size, :]
        else:
            prop_samp_size = props.shape[0]
        
        '''
        Expand properties and sample latent points
        '''
        # p_size = prop_samp_size
        # n_latent_prop_pairs = p_size * latent_sample_size

        # Expand props by latent_sample size
        properties = props.repeat_interleave(latent_sample_size, dim=0)
        # Sample latent points for each property
        latent_points = torch.randn(properties.shape[0], self.model.latent_dim, device=self.model.device, dtype=torch.float32)  * self.prior_std

        # exp_props is [batch_size x latent_sample_size , 1]
        # exp_latents is [batch_size x latent_sample_size , latent_size]

        '''
        2. Sample Decoder log likelyhoods and decoder actions
        '''
        
        decoder_log_probs, decoder_actions, expanded_properties = self._sample_decoder_actions_and_likelihoods(
                                                input_latent = latent_points, 
                                                input_props = properties, 
                                                decoder_sample_size=decoder_sample_size)
        # decoder_log_probs is shape 2560
        # decoder actions is shape 2560 x 101

        '''
        3. Compute log likelyhoods under LSTM
        '''

        with torch.no_grad():
            # We will set the initial token
            max_rules = max_seq_len - 1

            # !!! LSTM expects initial actions in the input, returns outputs without initial actions
            start_tokens = torch.ones((decoder_actions.shape[0], 1), dtype=torch.int, device=self.model.device)

            # Concatenate the ones tensor with the original actions tensor along dim=1
            lstm_input_actions = torch.cat((start_tokens, decoder_actions), dim=1)

            # lstm_input_actions is  2560 x 102 (added 1)

            hidden = self.reg_model.init_hidden(expanded_properties.shape[0], device)
            # returns outputs, hidden

            # decoeder actions is lots x 102 x 47
            lstm_logits = self.reg_model.forward(x=lstm_input_actions,
                                                            properties=expanded_properties,
                                                            hidden = hidden,
                                                            use_teacher_forcing = True, 
                                                            sampling = False,
                                                            return_actions = False, # UGLY: Doesn't matter 
                                                            return_both = False,
                                                            seq_len = decoder_actions.shape[1] + 1)[0]

            lstm_log_probs = self.log_probs_from_logits(actions=decoder_actions, logits=lstm_logits)

        '''
        4. Compute forward pass of regularizer
        '''
        if old_version:
            preg_loss = (-1) * ((lstm_log_probs) * decoder_log_probs).mean()
        else:
            preg_loss = PolicyRegularizer.apply(decoder_log_probs, lstm_log_probs).mean()

        '''
        5. Average out to get final loss, weight and return
        '''

        return self.regularizer_weight * preg_loss


        
    '''
    def compute_policy_regularizer_old(self, props, prop_samp_size = None, latent_sample_size = 4, decoder_sample_size = 10):
        
        #props <-- batch_size number of properties
        #prop_sample_size <-- number of properties to sample
        #latent_sample_size <-- number of latent points to sample per property
        #decoder_sample_size <-- number of 

        #TODO: Find way to get more properties if needed, by default I'll fix the number of properties to sample to the batch size
        #TODO: Amina mentioned I need to write the bw pass for this, perhaps my way doesn't work after all
        

        # 1. Compute sample of points from decoder
        #       a. sample latent points, latent_sample_size for each y
        if prop_samp_size == None:
            prop_samp_size = props.shape[0]
        # assert prop_samp_size <= props.shape[0]

        props = props[:prop_samp_size, :]
        p_size = prop_samp_size # 64
        n_latent_prop_pairs = p_size * latent_sample_size

        #       Properties repeated by latent_sample size
        properties = props.repeat_interleave(latent_sample_size, dim=0)

        # props is b_size x 1 | 64 x 1
        # properties is (b_size * latent_sample_size) x 1 | 128 x 1
        # latent_points is (p_size * latent_sample_size) x latent_dim_size | 128 x 56

        # Sample latent points for properties, at this point we've sampled latent_sample_size points for each property value
        # latent_points = np.random.normal(0, self.model.eps_std, size=(n_latent_prop_pairs, self.model.latent_dim))
        # latent_points = torch.tensor(latent_points, dtype=torch.float32)
        latent_points = torch.randn(n_latent_prop_pairs, self.model.latent_dim, device=self.model.device, dtype=torch.float32)

        # we now sample from the decoder decoder_sample_size times for each latent-property pair
        # decoder_actions = torch.zeros((n_latent_prop_pairs * decoder_sample_size, self.model.max_decode_steps) ,dtype=torch.int64, device=self.model.device)

        total_sample_size = n_latent_prop_pairs * decoder_sample_size
        # torch.Size([640, 101])

        # decoder_actions should be [total samples of from decoder] x seq_length = (p_size * z_per_p * x_per_zp_pair) x max_decode_steps
        # = (n_latent_prop_pairs * decoder_sample_size) x max_decode_steps = 128 x 101
        # (64 * 2 * 5) x 101 = 640 x 101
        
        # we now sample from the decoder decoder_sample_size times for each latent-property pair
        all_decoder_actions = torch.zeros((total_sample_size, self.model.max_decode_steps) ,dtype=torch.int64)  

        # LSTM negative log likelyhoods -- what we want to minimize
        # lstm_log_probs = torch.zeros( [total_sample_size], dtype=torch.float32, device=self.model.device)
        # decoder_log_probs = torch.zeros( [total_sample_size], dtype=torch.float32, device=self.model.device)

        decoder_log_probs, decoder_actions = self._sample_decoder_actions_and_likelihoods(input_latent = latent_points, input_props = properties, decoder_sample_size=decoder_sample_size)

        # Fetch LSTM log probs
        
        expanded_properties =  properties.repeat_interleave(decoder_sample_size, dim=0)
        # Expand props
        lstm_log_probs = self._compute_mol_log_probs_under_lstm(decoder_actions, expanded_properties)

        # Both log probabilities should be in range log([0, 1]) -> (-inf, 0]


        # both lstm_log_probs and decoder_log_probs are [n_pairs * decoder_sample_size] | [32] 

        lstm_log_probs = (-1 * lstm_log_probs).detach()

        # Should be in range [0, inf)
        regularizer_scores = lstm_log_probs * decoder_log_probs

        # regularizer_loss_unbatched = lstm_log_probs
        regularizer_loss = torch.mean(regularizer_scores)

        return regularizer_loss
    '''


    def compute_dkl_regularizer_loss(self, props, p_theta_avg = 1):
        '''
        Loss is sum of KLD's between next-token probabilities for decoder and lstm under the LSTM actions, and respecting LSTM end token
        p_theta_avg is how many p_\theta samples we average over before the KL_D
        '''

        '''
        Fetch decoder logits and convert to probabilities
        '''

        # Expand properties by number of latent samples 
        props_repeated = props.repeat_interleave(p_theta_avg, dim=0)  

        # sample latent points
        latent_points_expanded = torch.randn(props_repeated.shape[0], self.model.latent_dim, dtype=torch.float32, device=self.device) * self.prior_std # * self.model.eps_std

        '''
        Fetch LSTM logits and actions and convert to probabilities
        '''
        with torch.no_grad():
            hidden = self.reg_model.init_hidden(props.shape[0], self.device)

            lstm_actions, lstm_logits = self.reg_model.forward(x=None, 
                                                            properties=props,
                                                            hidden = hidden,
                                                            use_teacher_forcing = False, 
                                                            sampling = True,
                                                            return_actions = False,
                                                            return_both = True,
                                                            seq_len = self.model.max_decode_steps + 1)
            # Might be unnecessary?
            lstm_actions = lstm_actions.detach()
            lstm_logits = lstm_logits.detach()

        # remove max for stability
        lstm_logits_stable = lstm_logits - lstm_logits.max(dim=-1, keepdim=True)[0]
        
        lstm_probs = torch.softmax(lstm_logits_stable, dim=-1)

        # GET DECODER LOGITS ETC...
        decoder_logits = self.model.state_decoder(
            z=latent_points_expanded, 
            y=props_repeated).permute(1, 0, 2)

        # subtract max and convert to probs
        stable_decoder_logits = decoder_logits - decoder_logits.max(dim=-1, keepdim=True)[0]

        decoder_probs_exp = torch.softmax(stable_decoder_logits, dim=-1)

        # Average each group of n probabilities

        # decoder log probs is b_size * p_theta_avg x seq_len x decision_dim

        reshaped_tensor = decoder_probs_exp.view(props.shape[0], p_theta_avg, lstm_probs.shape[-2], lstm_probs.shape[-1])
 
        decoder_probs_avg = torch.mean(reshaped_tensor, dim=1)

        # lstm_probs is 64 x 101 x 47
        # decoder_probs is 64 x 101 x 47
        # lstm_actions is 64 x 101

        '''
        Compute KLD between the two
        '''

        kl_divs = F.kl_div(decoder_probs_avg.log(), lstm_probs, reduction='none').sum(dim=2)

        #kl_divs is b_size x seq_len

        '''
        Mask out probs after end token
        '''

        end_token_mask = (lstm_actions == self.sd.char_idx[self.sd.END])
        
        cumulative_end_mask = end_token_mask.cumsum(dim=1) >= 1
        
        first_end_token_indices = end_token_mask.int().argmax(dim=1)

        row_indices = torch.arange(lstm_actions.shape[1], device=lstm_actions.device).unsqueeze(0).expand(lstm_actions.shape[0], -1)
        cumulative_end_mask = row_indices > first_end_token_indices.unsqueeze(1)

        selected_kl_divs = kl_divs.masked_fill(cumulative_end_mask, 0)

        '''
        Sum weight and return
        '''

        return selected_kl_divs.mean() * self.regularizer_weight


    def _save_model(self, base_dir, info, epoch_no, valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'SD_REG_VANILLA_VAE_{info}_Epoch_{epoch_no}_Vl_{valid_loss:.3f}'
        print(f'Saving: {base_name} to {base_dir}')
        save_model(self.model, base_dir, base_name)


    def save_params(self, run_name=None, save_dir = None):
        if save_dir is None:
            save_dir = self.model_save_dir  # Use the directory specified in the trainer's constructor
        
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Define the file path for saving the parameters
        params_file_path = os.path.join(save_dir, f'{run_name}_training_params.json')
        
        # Dictionary of parameters to save
        params = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'property_names':list(self.property_names),
            'property_normalizations_means': list(self.pnorm_means),
            'property_normalization_stds':list(self.pnorm_stds),
            'device':self.device,
            'eps_std':self.eps_std,
            'beta':self.beta,
            'vocab_size': self.vocab_size,  # Assuming input_size is accessible in your model
            'max_seq_len': self.max_seq_len,  # Assuming max_rules corresponds to max sequence length in your model
            'save_every_n_val_cycles': self.save_every_n_val_cycles,
            'early_stopping_every_epochs': self.early_stopping_epochs
        }
        
        # Write the parameters dictionary to a JSON file
        with open(params_file_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"Training parameters saved to {params_file_path}")

    def fit(self):
        '''
        Complete training run of model
        '''
        # TODO: Early stopping 
        haikunator = Haikunator()
        run_id = haikunator.haikunate(token_length=2)
        # Save run parameters
        self.save_params(run_name=run_id)

        print(f'Training run ID: {run_id}')

        best_val_loss = np.inf
        best_val_epoch = 0
        # Save training parameters
        # TODO: Implement
        for epoch_no in range(1, self.max_epochs + 1):

            ## Go go gadget beta annealing
            self.beta = self.beta_annealing_function(self.beta, epoch_no)
            self.model.beta = self.beta

            self.regularizer_weight = self.reg_weight_annealing_function(self.regularizer_weight, epoch_no)

            # Train for epoch
            torch.cuda.empty_cache()
            avg_loss_total_epoch, avg_loss_recon_epoch, avg_loss_kl_epoch, avg_loss_reg_epoch = self.run_epoch_vanilla_vae('train', epoch=epoch_no)

            # Run validation
            with torch.no_grad():
                avg_loss_total_val, avg_loss_recon_val, avg_loss_kl_val, avg_loss_reg_val = self.run_epoch_vanilla_vae('valid', epoch=epoch_no)

                self.lr_scheduler.step(avg_loss_total_val)

            # wandb logging
            if self.wandb_project != None:
                wandb.log({
                    'Epoch No':epoch_no,
                    'Train Loss Total':avg_loss_total_epoch,
                    'Train Loss Recon':avg_loss_recon_epoch,
                    'Train Loss KL Divergence':avg_loss_kl_epoch,
                    'Train Loss Regularizer':avg_loss_reg_epoch,
                    'Validation Loss Total':avg_loss_total_val,
                    'Validation Loss Recon':avg_loss_recon_val,
                    'Validation Loss KL Divergence':avg_loss_kl_val,
                    'Validation Loss Regularizer':avg_loss_reg_val,
                }, step = epoch_no)

            # See if val loss improved
            if avg_loss_total_val < best_val_loss:
                print(f'Achieved Best Valid Loss so Far: {avg_loss_total_val}')
                best_val_loss = avg_loss_total_val
                best_val_epoch = epoch_no

            # Save model params If val better or save every n validations
            if avg_loss_total_val <= best_val_loss or epoch_no % self.save_every_n_val_cycles == 0:
                self._save_model( self.model_save_dir, run_id, epoch_no, avg_loss_total_val)

            if epoch_no - best_val_epoch > self.early_stopping_epochs:
                print('*' * 20)
                print(f'Early Stopping on Epoch {epoch_no}. No validation improvement in {self.early_stopping_epochs} epochs')
                print('*' * 20)
                return

    def log_probs_from_logits_iterative(self, actions, logits):
        '''
        returns the log probability of batch of sequences represented by batch of logits according to batch of selected actions

        actions is tensor of ints batch_size x seq_len
        logits is tensor of floats batch_size x seq_len x decision_dim

        Log(p_1 x p_2 x ... x p_n) = log p_1 + log p_2 + ... + log p_n

        Should only consider probabilities up untill first occurance of end token sd.END
        '''

        raise Exception('WARNING: current implementation modifies in place')

        batch_size, seq_len, decision_dim = logits.size()

        has_not_ended = torch.ones(batch_size, device=logits.device)

        log_probs = torch.zeros(batch_size, device=logits.device)
        # Walk through the sequence
        for i in range(seq_len):
            # If has_ended for that item in the batch, should add zero
            ns_logits = logits[:, i, :] - logits[:, i, :].max()
            log_probs += (has_not_ended) * (F.log_softmax(ns_logits, dim=-1).gather(1, actions[:, i].unsqueeze(1)).squeeze())

            # Update has_ended, now probabilities following the ended actions shouldn't be considered
            has_not_ended[actions[:, i] == self.sd.char_idx[self.sd.END]] = 0


        return log_probs

    def log_probs_from_logits(self, actions, logits, weight = False):
        '''
        Weight to avoid bias towards shorter strings with higher probs
        '''
        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # For numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
        
        end_token_mask = (actions == self.sd.char_idx[self.sd.END])
        cumulative_end_mask = end_token_mask.cumsum(dim=1) >= 1
        first_end_token_indices = end_token_mask.int().argmax(dim=1)
        batch_size, seq_len = actions.shape
        row_indices = torch.arange(seq_len, device=actions.device).unsqueeze(0).expand(batch_size, -1)
        cumulative_end_mask = row_indices > first_end_token_indices.unsqueeze(1)

        '''
        Currently cumulative end mask if false up untill the first occurance of end token
        I need it to also be false exactly for the first occurance of end token
        '''
        
        selected_log_probs = selected_log_probs.masked_fill(cumulative_end_mask, 0)
        
        total_log_probs = selected_log_probs.sum(dim=-1)

        if weight:
            # Get lengths of strings before end action | number of occurances of False in cumulative_end_mask
            counts = (cumulative_end_mask == False).sum()

            # device total_log_probs by respective string lengths
            return total_log_probs / counts
        else:
            return total_log_probs

        '''
        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # For numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
        end_token_mask = (actions == self.sd.char_idx[self.sd.END])
        cumulative_end_mask = end_token_mask.cumsum(dim=1) > 1
        # This: selected_log_probs[cumulative_end_mask] = 0

        print(actions)
        print(cumulative_end_mask)
        
        selected_log_probs = selected_log_probs.masked_fill(cumulative_end_mask, 0)
        
        total_log_probs = selected_log_probs.sum(dim=-1)

        return total_log_probs
        '''


    def compute_policy_2_regularizer(self, props, n_lstm_samples = 5):
        '''
        Compute second version of policy regularizer

        props -> property scores 
        n_lstm_samples -> number of samples per property score to sample from the LSTM
            ! sample a seperate latent point for each of these

            TODO: Check I'm sampling from lstm for 1 y and 1 z
        '''

        '''
        Repeat properties by n_lstm_samples and sample latent points
        '''

        '''
        IS THIS WRONG? SHOULD IT BE THE SAME Y AND Z. HERE ITS FOR THE SAME Y SAMPLE Z AND LSTM
        '''
        expanded_props = props.repeat_interleave(n_lstm_samples, dim=0)
        
        latent_points = torch.randn(expanded_props.shape[0], self.model.latent_dim, device=self.model.device, dtype=torch.float32) * self.prior_std # * self.model.eps_std
        # latent_points is shape [b_size * n_lstm_samples x self.latent_dim]

        '''
        Sample lstm actions
        '''
        # expanded_props is shape [b_size * n_lstm_samples x 1]

        hidden = self.reg_model.init_hidden(expanded_props.shape[0], self.model.device)

        with torch.no_grad():
            lstm_actions, lstm_logits = self.reg_model.forward(x=None,
                                                                properties=expanded_props,
                                                                hidden = hidden,
                                                                use_teacher_forcing = False,
                                                                sampling = True,
                                                                return_actions = False,
                                                                return_both = True)
            lstm_actions = lstm_actions[:, :self.model.max_decode_steps - 2].detach()

        # Decoder does not predict start token so don't do this:
        # prepend_tokens = torch.full((expanded_props.shape[0], 1), self.sd.char_idx[self.sd.BEGIN], dtype=lstm_actions.dtype, device=lstm_actions.device
        # Concatenate the prepend token tensor with the original actions tensor along dimension 1 (seq_len)
        # lstm_actions = torch.cat((prepend_tokens, lstm_actions), dim=1)

        # lstm_logits adter adding start token is [b_size * n_lstm_samples x (seq_len - 1) x decision_dim]
        # !Does not contain start token

        '''
        Sample Decoder Actions
        '''

        decoder_logits = self.model.state_decoder(z=latent_points, y=expanded_props).permute([1, 0, 2])

        # Decoder logits is [b_size * n_lstm_samples, seq_len + 1, latent_dim]
        # decoder logits is predicting the tokens from start token onwards
        decoder_logits = decoder_logits[:, :-2, :]
        # prune two tokens to ensure compatible length 
        # TODO: Fix this so if someone else sets the seq len parameter they won't accidentally prune tokens 

        # Decoder logits is [b_size * n_lstm_samples, seq_len - 1, latent_dim]
        
        # Compute decoder log probs under each latent property pair

        '''
        logits = torch.tensor([
            [
            [0.3, 0.0, 0.3, 0.0],  # Logits for token at index 0 (equal prob for simplicity)
            [0.0, 2.0, 0.0, 0.0],  # Logits for token at index 1 (high prob for index 1)
            [0.0, 0.0, 3.0, 0.0],  # Logits for token at index 2 (high prob for index 2, END token)
            [0.0, 0.0, 0.0, 4.0],  # Logits for token at index 3 (high prob for index 3)
            [1.0, 0.0, 0.0, 0.0]   # Logits for token at index 4 (high prob for index 0)
                ]
            ])

        actions  = torch.tensor([[1, 2, 2, 2, 1]])

        print(self._compute_mol_log_probs_under_decoder(actions, logits))
        print(self.log_probs_from_logits(actions[:,1:], logits[:,1:,:]))
        print(self.log_probs_from_logits_iterative(actions[:,1:], logits[:,1:,:]))
        '''

        '''
        Compute Log Probs of lstm sample under decoder
        '''
        log_probs = self.log_probs_from_logits(actions= lstm_actions, logits=decoder_logits)

        # We're minimizing the negative of this averaged over the batch (loss)
        return self.regularizer_weight * -1 * log_probs.mean()

    def alternate_policy_2_regularizer(self, props, n_lstm_samples = 60):
        # Expand props by number of lstm samples
        props_ext = torch.repeat_interleave(props, n_lstm_samples, dim=0)


        '''#  4 unique latent + prop pair for each LSTM Sample 
        latent_points = torch.randn(
                    props_ext.shape[0], 
                    self.model.latent_dim, 
                    device=self.model.device,
                    dtype=torch.float32) * self.prior_std
        '''
        # n_samples lstm samples for each LATENT PROP pair
        latent_points = torch.randn(
                    props.shape[0],
                    self.model.latent_dim, 
                    device=self.model.device,
                    dtype=torch.float32) * self.prior_std

        latent_points = torch.repeat_interleave(latent_points, n_lstm_samples, dim=0)
        

        # Sample from decoder
        with torch.no_grad():
            '''
            start_tokens = torch.ones(
                                (props_ext.shape[0], 1),
                                dtype=torch.int,
                                device = self.model.device
                                )
            '''

            hidden = self.reg_model.init_hidden(props_ext.shape[0], self.model.device)

            lstm_actions, lstm_logits = self.reg_model.forward(
                                                x = None,
                                                properties = props_ext,
                                                hidden = hidden,
                                                use_teacher_forcing = False,
                                                sampling = True, # As opposed to argmax
                                                return_actions = True,
                                                return_both = True,
                                                seq_len = self.model.max_decode_steps + 1
                                            )
            '''
            Logits: 80 x 37 x 47 | No start token / action
            Actions: 80 x 37 | No Start token / action
            '''

        ''' 
        Latent Points: 80 x 56
        '''

        decoder_logits = self.model.state_decoder(latent_points, props_ext).transpose(0, 1)
        '''
        Decoder Logits: 80 x 37 x 47 | No start token
        '''
        # Convert decoder logits to probabilies
        decoder_probs = F.softmax(decoder_logits, dim=2)
        
        # Select decoder logits corresponding to actions
        # NOTE: Actions[i] is picked from logits[i]

        # unsqueeze makes lstm_actions 80 x 37 x 1
        # Ggather takes input - dim - index, so it will take each action and select the appropiate logit
        # resqueeze to get remove extra dim
        selected_probs = torch.gather(decoder_probs, 2, lstm_actions.unsqueeze(-1)).squeeze(-1)

        '''
        Selected Logits is 80 x 37
        '''
        # Mask out logits post first occurance of end token
        mask = (lstm_actions == self.sd.char_idx[self.sd.END]).cumsum(dim=1) == 1
        mask = mask.int()
        indices_of_first_twos = mask.argmax(dim=1)
        # indices_of_first_twos  80 | contains literal index of first twos

        expand_indices = indices_of_first_twos[:, None]
        range_indices = torch.arange(lstm_actions.shape[1], device=self.device)[None, :]
        valid_mask = range_indices <= expand_indices
        # Valid mask contains True's up until and including first end token

        '''
        Instead of maximizing the expected probability
        We will minimize the expected log probability
        '''

        # Decoder logits from outside the mask should be set to 1.0
        log_selected_probs = torch.log(selected_probs)

        masked_decoder_log_probs = log_selected_probs.masked_fill(mask=(~valid_mask), value=(0.0))

        sample_probs = torch.sum(masked_decoder_log_probs, dim=1)

        return self.regularizer_weight * -1 * sample_probs.mean()


    def run_epoch_vanilla_vae(self, phase, epoch = None):
        if phase == 'train':
            print(f'Training epoch {epoch}')
            self.model.train()
            self.model.reparam = True
            data_loader = self.train_data_loader
        else:
            print('Running Validation:')
            self.model.eval()
            self.model.reparam = False
            data_loader = self.val_data_loader

        # TODO: Progress bar
        n_batches = len(data_loader.dataset) // data_loader.batch_size + (len(data_loader.dataset) % data_loader.batch_size != 0)
        pbar = tqdm(range(0, n_batches), unit='batch')

        total_batches = 0

        # Total losses for epoch
        total_epoch_recon = 0
        total_epoch_kl = 0
        total_epoch_reg = 0

        for bindex, (inp, tgt, prop) in enumerate(data_loader):
            # Send everything to device
            inp, tgt, prop = inp.to(self.device), tgt.to(self.device), prop.to(self.device)
            # assert inp.shape[0] == prop.shape[0]
            # inp.shape - batch_size x max_seq_len
            # tgt.shape - batch_size x max_seq_len
            # prop.shape - batch_size x 1 ??

            # Inp contains start token
            # tgt does is offset by one

            if self.unconditional:
                prop = torch.zeros_like(prop)

            recon_loss, kl_loss, raw_logits = self.model.forward(x_inputs=inp, y_inputs=prop, true_binary=tgt,  return_logits=True)
            
            regularizer_loss = torch.tensor(0.0, device=self.device)

            if self.regularizer_type == 'KLD':
                for i in range(self.num_reg_samples):
                    regularizer_loss += self.compute_dkl_regularizer_loss(prop)
                regularizer_loss /= self.num_reg_samples
            elif self.regularizer_type == 'Pol':
                # regularizer_loss = self.compute_policy_regularizer(props= prop, latent_sample_size = self.preg_latent_sample_size, decoder_sample_size = self.preg_decoder_sample_size)
                regularizer_loss = self.alternate_policy_regularizer(props = prop, latent_sample_size = self.preg_latent_sample_size, decoder_sample_size = self.preg_decoder_sample_size)
            elif self.regularizer_type == 'Pol2':
                # regularizer_loss = self.compute_policy_2_regularizer(props = prop, n_lstm_samples=self.n_lstm_samples)
                regularizer_loss = self.alternate_policy_2_regularizer(props = prop, n_lstm_samples=self.n_lstm_samples)


            batch_loss = recon_loss + kl_loss + regularizer_loss

            # We therefore expect batch_loss to be of dim batch_size x 1
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_batches += 1
            batch_loss = batch_loss.detach() 
            recon_loss = recon_loss.detach()
            kl_loss = kl_loss.detach()
            regularizer_loss = regularizer_loss.detach()


            total_epoch_kl += kl_loss
            total_epoch_recon += recon_loss
            total_epoch_reg += regularizer_loss


            # For each batch
            if total_batches % 20 == 0:
                if phase == 'train':
                    pbar.set_description(f'Epoch: {epoch} | Recon: {recon_loss} - KL {kl_loss}  - Reg Loss {regularizer_loss} - Total {batch_loss}')
                else:
                    pbar.set_description(f'Validation Lossses | Recon: {recon_loss} - KL {kl_loss} - Reg Loss {regularizer_loss}- Total {batch_loss}')
            pbar.update(1)

        pbar.close()
        avg_loss_reg = total_epoch_reg / total_batches
        avg_loss_recon = total_epoch_recon / total_batches
        avg_loss_kl = total_epoch_kl / total_batches
        avg_loss_total = avg_loss_kl + avg_loss_recon + avg_loss_reg

        if phase == 'train':
            print(f' --- Epoch {epoch} Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - Avg Reg {avg_loss_reg} | AVG Total {avg_loss_total}')
        else:
            print(f'--- Validation Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - Avg Reg {avg_loss_reg} | AVG Total {avg_loss_total}')
        # Report final loss
        # TODO: Return total avg loss in epoch
        return avg_loss_total, avg_loss_recon, avg_loss_kl, avg_loss_reg


if __name__ == "__main__":
        '''
        RUN PARAMETERS
        '''
        dset = 'QM9_36' # 'ZINC' # or 'QM9' 'QM9_36' 'ZINC120'
        device = 'cpu'
        regularizer_type = None # 'Pol' # None # 'KLD' # 'Pol' # in 'Pol', 'Pol2', 'KLD', None
        run_save_folder = '../../GB-QM9/'
        
        if regularizer_type == 'KLD':
            reg_weight = 0.1
        else:
            reg_weight = 0.05

        latent_dim = 156 # 56
        batch_size = 16 # 128 # 265
        run_name = ''

        beta = 0.001
        eps_std = 1.0

        preg_decoder_sample_size = 20
        preg_latent_sample_size = 4
        prior_std = 1.0
        # Is weighted flag set correctly
        # Is unconditional flag set correctly
        '''
        ANNEALING FUNCTION
        '''
        beta_anneal_funct = None # Identity
        reg_anneal_fnct = None

        # 100 Start 200 End Linear increase up from 0.01 to 1
        def linear_anneal(beta, epoch):
            initial_beta = 0.01
            max_beta = 1.0
            start_epoch = 100
            end_epoch = 200
            if epoch < start_epoch:
                return initial_beta
            elif epoch <= end_epoch:
                return initial_beta + (max_beta - initial_beta) * (epoch - start_epoch) / (end_epoch - start_epoch)
            else:
                max_beta

        def anneal_beta(beta, epoch, total_epochs=120, start_epoch=29, max_val=1.0):
            if epoch < start_epoch:
                return beta
            else:
                progress = (epoch - start_epoch) / (total_epochs - start_epoch)
                return max_val / (1 + math.exp(-12 * (progress - 0.5)))

        beta_anneal_funct = anneal_beta

        '''
        /RUN PARAMETERS
        '''
        run_name += 'cVAE'
        if dset == 'QM9_36':
                data_path = '../data/QM9_36/'
                max_seq_len = 37
                run_name += '-QM936'
        elif dset == 'ZINC120':
                data_path = '../data/ZINC120/'
                max_seq_len = 121
                run_name += '-ZINC120'
        elif dset == 'ZINC':
                data_path = '../data/ZINC250K/'
                max_seq_len = 279
                run_name += '-ZINC'
        elif dset == 'QM9':
                data_path = '../data/QM9/'
                max_seq_len = 37
                run_name += '-QM9'

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_y, valid_x_enc, valid_y = load_smiles_and_properties(data_path)
        # If you want wandb to be offline, need to set environ variable


        if 'ZINC' in dset:
             reg_model_path = './reg_models/LSTM_ZINC120_46_0.461.pt'
             reg_param_path = './reg_models/LSTM_ZINC120_46_0.461.json'
        if 'QM9' in dset:
             reg_model_path = './reg_models/LSTM_20_1.190.pt'
             reg_param_path = './reg_models/LSTM_20_1.190.json'

        # Load Pretrained Model
        if regularizer_type in ['KLD', 'Pol', 'Pol2']:
            # Pretrained Model
            if dset == 'QM9':
                prior_model_path = './pretrained_models/CVAE_Epoch_80_Vl_0.375'
            if dset == 'QM9_36':
                prior_model_path = './pretrained_models/CVAE_Epoch_80_37_INVERTED_Vl0337'
                prior_model = None
            if 'ZINC' in dset:
                prior_model_path = None
                prior_model = None
            else:
                '''
                prior_model = load_model(model_class=VanillaMolVAE, 
                                    model_definition=prior_model_path + '.json',
                                    model_weights=prior_model_path + '.pt',
                                    device=device)

                 prior_model = prior_model.to(device)
                 prior_model.device = device
                 prior_model.encoder.device = device
                 prior_model.state_decoder.device = device
                '''
                prior_model = None

            n_lstm_samples = 50
            if regularizer_type == 'KLD':
                # reg_weight = 0.01
                run_name += '-KLD'
            elif regularizer_type == 'Pol2':
                run_name += '-Pol2'
            else:
                reg_weight = 0.01
                run_name += '-Pol'

        else:
            # reg_weight = 0.0
            n_lstm_samples = 0
            prior_model = None

        ###############
        '''
        prior_model_path = ''
        prior_model = load_model(model_class=VanillaMolVAE, 
                            model_definition=prior_model_path + '.json',
                            model_weights=prior_model_path + '.pt',
                            device=device)

        prior_model = prior_model.to(device)
        prior_model.device = device
        prior_model.encoder.device = device
        prior_model.state_decoder.device = device
        '''
        ###############

        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = (run_save_folder + run_name)

        # Instantiate Trainer Class
        trainer =  VanillaVAETrainerReg(
                regularizer_type=regularizer_type,
                regularizer_model_path = reg_model_path, # './reg_models/LSTM_12_1.180.pt', # '../models/LSTM_QM9/batch_size_64_2/LSTM_20_1.190.pt',
                regularizer_param_path = reg_param_path, # './reg_models/LSTM_12_1.180.json', # '../models/LSTM_QM9/batch_size_64_2/LSTM_20_1.190.json',
                regularizer_weight = reg_weight,
                train_x_enc=train_x_enc,
                train_y=train_y,
                valid_x_enc=valid_x_enc,
                valid_y=valid_y,
                property_names=["LogP"],
                device=device,
                batch_size=batch_size,
                latent_dim= latent_dim,
                beta=beta,
                max_seq_len=max_seq_len,
                eps_std=eps_std,
                model=prior_model,
                decoder_embedding_dim=47,
                learning_rate = 1e-3,
                model_save_dir = (run_save_folder + run_name),
                save_every_n_val_cycles = 3,
                max_epochs = 500,
                wandb_project = run_name,
                early_stopping_epochs = 50,
                preg_latent_sample_size = preg_latent_sample_size,
                preg_decoder_sample_size = preg_decoder_sample_size,
                prior_std=prior_std,
                beta_annealing_function=beta_anneal_funct,
                reg_weight_annealing_function=reg_anneal_fnct,
                n_lstm_samples=n_lstm_samples)

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, valid_x_enc, train_y, valid_y

        trainer.fit()

