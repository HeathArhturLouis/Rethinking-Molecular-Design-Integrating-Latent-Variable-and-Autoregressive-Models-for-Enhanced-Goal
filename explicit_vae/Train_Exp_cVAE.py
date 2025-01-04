import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import wandb

import sys

import os

import numpy as np
from tqdm import tqdm

sys.path.append('../utils/')
from rnn_utils import load_smiles_and_properties, get_tensor_dataset, save_model, load_model, rnn_start_token_vector

from reg_rnn_model import ConditionalSmilesRnn

from smiles_char_dict import SmilesCharDictionary

from haikunator import Haikunator
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


from model_vanilla_vae import VanillaMolVAE

from torch.distributions import Categorical

import random

import math

N_WORKERS_DL = 4


class PolicyRegularizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs_decoder, log_probs_lstm):
        # ctx.save_for_backward(log_probs_lstm, log_probs_decoder)
        ctx.save_for_backward(log_probs_lstm)

        return (-1 * log_probs_lstm) # The mean of this when returned will be our final loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        TODO: Seems there's no need to store log_probs_decoder
        '''
        # log_probs_lstm is [len(x_sample)]
        # grad_output is [len(x_sample)]

        log_probs_lstm, = ctx.saved_tensors #, log_probs_decoder = ctx.saved_tensors

        weighted_grads = grad_output * log_probs_lstm

        # Return None for x_inputs and gradients for decoder parameters
        return (-1 * weighted_grads) , None



class RegularizedVAETrainer:
    def __init__(self,
                 regularizer_type, # in [None, 'KLD', 'Policy']
                 regularizer_model_path, # Path to regularizer parameters
                 regularizer_weight, # Coefficient for weighting regularizer loss 
                 train_x_enc, 
                 train_y, 
                 valid_x_enc, 
                 valid_y, 
                 property_names,
                 model, 
                 device, 
                 batch_size,
                 latent_dim=56,
                 beta=1.0,
                 max_seq_len=100,
                 eps_std=0.01,
                 decoder_embedding_dim=47, # Prev was tok size
                 learning_rate = 1e-3,  model_save_dir = '../models/VANILLA_VAE_REGULARIZED/', 
                 save_every_n_val_cycles = 3, max_epochs = 500,
                 teacher_forcing_prob = 0.1,
                 wandb_project = None,
                 early_stopping = 15,
                 prior_std=1.0,
                 beta_annealing_function=None):

        ''' If true sets conditional information to zero for measuring unconditional performance '''
        if beta_annealing_function is None:
            self.beta_annealing_function = (lambda beta, epoch : beta) 
        else:
            self.beta_annealing_function = beta_annealing_function

        self.unconditional = False

        self.prior_std = prior_std

        assert regularizer_type in [None, 'KLD', 'Pol', 'Pol2']
        self.regularizer_type = regularizer_type

        if regularizer_type is None:
            self.regularizer_param_path = None
            self.regularizer_weight = None
            self.reg_model = None
        else:
            self.regularizer_param_path = regularizer_model_path
            self.regularizer_weight = regularizer_weight

            # Load Regularizer model
            self.reg_model = load_model(model_class=ConditionalSmilesRnn, 
                                    model_definition=regularizer_model_path + '.json', 
                                    model_weights=regularizer_model_path + '.pt', 
                                    device=device) # Should be on device
            self.reg_model = self.reg_model.eval()


        # Number of epochs without val loss improvement before training terminates
        self.early_stopping = early_stopping

        # Regularizer Training Params
        self.teacher_forcing_prob = teacher_forcing_prob

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
                            decoder_mod_type = 'gru',
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
        # self.reg_optimizer = torch.optim.Adam(self.model.state_decoder.parameters())

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

        self.param_dict = {
                'regularizer_type':self.regularizer_type,
                'regularizer_param_path':self.regularizer_param_path,
                'regularizer_weight':self.regularizer_weight,
                'property_names':list(self.property_names),
                'property_normalizations_means': list(self.pnorm_means),
                'property_normalization_stds':list(self.pnorm_stds),
                'device':device,
                'batch_size':self.batch_size,
                'latent_dim':latent_dim,
                'max_seq_len':max_seq_len,
                'beta':beta,
                'eps_std':eps_std,
                'decoder_embedding_dim':decoder_embedding_dim,
                'learning_rate':learning_rate,
                'model_save_dir':model_save_dir,
                'save_every_n_val_cycles':save_every_n_val_cycles,
                'max_epochs':max_epochs,
                'teacher_forcing_prob':teacher_forcing_prob,
                'wandb_project':wandb_project,
                'early_stopping':early_stopping,
                'regularizer_prior_std':self.prior_std
                }

        if self.wandb_project != None:
            wandb.init(project=self.wandb_project)
            wandb.config = self.param_dict
        
        self.sd = SmilesCharDictionary()


    def compute_regularizer_loss_kld(self, props, sample = True):
        '''
        Inputs:
        props -- b_size x 1 sized tensor of property scores
        sample -- if true sample
        '''
        
        device = self.model.device
        max_seq_len = self.model.max_decode_steps # 101
        decision_dim = self.model.decoder_embedding_dim # 47
        b_size = props.shape[0] # 64
        self.reg_model = self.reg_model # .to(self.device) <-- should already be on device


        props = props.to(self.device) # props is b_size x 1

        '''
        SAMPLE LSTM MODEL ACTIONS AND LOGITS
        Random Sampling, NO TF, make sure not to track gradients
        '''

        with torch.no_grad():
            # We will set the initial token
            max_rules = max_seq_len - 1

            hidden = self.reg_model.init_hidden(b_size, device)

            lstm_actions, lstm_outputs = self.reg_model.forward(x=None, 
                                                            properties=props, 
                                                            hidden = hidden,
                                                            use_teacher_forcing = False, 
                                                            sampling = True, 
                                                            return_actions = False, # UGLY: Doesn't matter 
                                                            return_both = True,
                                                            seq_len = 121)


        lstm_actions = lstm_actions[:, :self.model.max_decode_steps - 2]
        lstm_outputs = lstm_outputs[:, :self.model.max_decode_steps - 2, :]

        # WARNING: No start token
        # LSTM actions: b_size x (seq_len - 1) | 64 x 99
        # LSTM outputs: b_size x (seq_len - 1) x decision_dim | 64 x 99 x 47

        # COMPUTE LOCALLY NORMD PROBABILITIES FROM VAE DECODER FOR LSTM ACTIONS
        # Random Latent Point (marginalize over z), for property

        # We can do this by calling the state decoder forward, and teacher forcing on the LSTM actions
        
        # Sample random z according to our target latent distribution
        latent_points = torch.randn(b_size, 
                                    self.model.latent_dim, 
                                    dtype=torch.float32, 
                                    device=self.device) * self.prior_std# * self.model.eps_std

        # Add start tokens to lstm outputs
        start_tokens = torch.ones([b_size, 1], dtype=lstm_actions.dtype, device=lstm_actions.device)

        # This is incredibly ugly, and I should honestly fix this

        # lstm_actions is 64 x 99

        end_tokens = torch.zeros([b_size, 1], dtype=lstm_actions.dtype, device=lstm_actions.device)

        aug_actions = torch.cat([start_tokens, lstm_actions, end_tokens], dim=-1)
        # aug_actions is 64 x 100

        # Get the logits from state decoder
        # forward(self, z, y, x_inputs = None, teacher_forcing = True, return_logits = True)


        # Teacher forcing means it predicts the next LSTM steps at each step of the LSTM sequence
        decoder_logits = self.model.state_decoder.forward(z=latent_points, 
                                                y=props, 
                                                x_inputs = aug_actions, 
                                                teacher_forcing = True,
                                                return_logits = True)


        # decoder_logits is b_size x seq_len x decision_dim
        # remove initial token: b_size x seq_len - 1 x decision_dim
        decoder_logits = decoder_logits[:, 1:-1, :] # Prune last token added before. We're counting on the fact that molecules are not 100 long in QM9

        # lstm_logits is without initial token and: b_size x seq_len - 1 x decision_dim

        # get indecies of first occurance of end token in lstm_actions
        mask = (lstm_actions == 2).cumsum(dim=1) == 1
        mask = mask.int()
        indices_of_first_twos = mask.argmax(dim=1)

        # Create mask of valid indecies
        expand_indices = indices_of_first_twos[:, None]  # Expand dimensions for broadcasting
        # range_indices = torch.arange(lstm_actions.shape[1])[None, :]  # [1, 101] range
        range_indices = torch.arange(lstm_actions.shape[1], device=self.device)[None, :]
        valid_mask = range_indices <= expand_indices

        # TODO: is this redundant?
        # valid mask is 64 x 99 boolean mask of valid indecies
        masked_lstm_logits = lstm_outputs * valid_mask[..., None]
        masked_decoder_logits = decoder_logits * valid_mask[..., None]

        # Softmax is applied to convert logits to probabilities
        lstm_probs = F.softmax(masked_lstm_logits, dim=2)
        decoder_probs = F.softmax(masked_decoder_logits, dim=2)

        # compute kl divergences for all pairs of distributions
        kl_div = F.kl_div(decoder_probs.log(), lstm_probs, reduction='none').sum(dim=2)

        # kl_div is 64 x 99

        # Step 4: Compute average KL divergence where the mask is True
        kl_div_sum = kl_div[valid_mask].sum()
        total_valid = valid_mask.sum()

        # Devide by total number of valid terms where kl is non zero
        average_kl_div = kl_div_sum / total_valid if total_valid > 0 else torch.tensor(0.)

        return average_kl_div * self.regularizer_weight
        

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
        
        # Write the parameters dictionary to a JSON file
        with open(params_file_path, 'w') as f:
            json.dump(self.param_dict, f, indent=4)
        
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
            self.beta = self.beta_annealing_function(self.beta, epoch_no)
            self.model.beta = self.beta

            # Train for epoch
            torch.cuda.empty_cache()
            train_loss_total, train_loss_rec, train_loss_kl, train_loss_reg = self.run_epoch_vanilla_vae('train', epoch=epoch_no)

            with torch.no_grad():
                valid_loss_total, valid_loss_rec, valid_loss_kl, valid_loss_reg = self.run_epoch_vanilla_vae('valid', epoch=epoch_no)

                self.lr_scheduler.step(valid_loss_total)
                # If val better or save every n validations

            if not (self.wandb_project is None):
                wandb.log({
                    'Epoch No':epoch_no,
                    'Train Loss Total':train_loss_total,
                    'Train Loss Recon':train_loss_rec,
                    'Train Loss KL Divergence':train_loss_kl,
                    'Train Loss Regularizer':train_loss_reg,
                    'Validation Loss Total':valid_loss_total,
                    'Validation Loss Recon':valid_loss_rec,
                    'Validation Loss KL Divergence':valid_loss_kl,
                    'Validation Loss Regularizer':valid_loss_reg,
                }, step = epoch_no)
            
            if valid_loss_total <= best_val_loss or epoch_no % self.save_every_n_val_cycles == 0:
                
                self._save_model( self.model_save_dir, run_id, epoch_no, valid_loss_total)
        
            if valid_loss_total < best_val_loss:
                print(f'Achieved Best Valid Loss so Far: {valid_loss_total}')
                best_val_epoch = epoch_no
                best_val_loss = valid_loss_total
            
            if epoch_no - best_val_epoch > self.early_stopping:
                # Early termination, save last model
                print()
                print(f'EARLY STOPPING: {best_val_epoch} was best epoch.')
                print()                    
                return

    def compute_log_probs(self, actions, logits, stability=True):
        if stability:
            min_log_value = -1e9

        log_probabilities = F.log_softmax(logits, dim=-1)
        log_probabilities = torch.clamp(log_probabilities, min=min_log_value)

        selected_log_probs = torch.gather(log_probabilities, 2, actions.unsqueeze(-1)).squeeze(-1)

        mask = (actions == self.sd.char_idx[self.sd.END]).cumsum(dim=1) == 1
        mask = mask.int()
        indices_of_first_twos = mask.argmax(dim=1)
        expand_indices = indices_of_first_twos[:, None]
        range_indices = torch.arange(actions.shape[1], device=self.device)[None, :]
        valid_mask = range_indices <= expand_indices

        masked_selected_log_probs = selected_log_probs.masked_fill(mask=(~valid_mask), value=0.0)
        sample_log_probs = torch.sum(masked_selected_log_probs, dim=1)

        return sample_log_probs

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
        decoder_logits = self.model.state_decoder(
                                z=latent_points, 
                                y=props_exp, 
                                x_inputs=None, 
                                teacher_forcing = False,
                                return_logits=True,
                                return_both=False) # .transpose(1, 0)

        decoder_logits = decoder_logits[:, 1:, :]
        
        '''
        Decoder logits is 64 x 37 x 47
        Decoder logits contains start token
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

            lstm_logits, new_hidden = self.reg_model.forward(
                                                x = input_actions,
                                                properties = props_exp_exp,
                                                hidden = hidden,
                                                use_teacher_forcing = True,
                                                sampling = True, # As opposed to argmax
                                                return_actions = False,
                                                return_both = False,
                                                seq_len = self.model.max_decode_steps
                                            )


        lstm_log_probs = self.compute_log_probs(actions = decoder_actions, logits = lstm_logits)

        loss = -1 * (lstm_log_probs * decoder_log_probs).mean() * self.regularizer_weight + 1.6
        
        # Original Version:
        # return (lstm_log_probs * decoder_log_probs).mean() * -1 * self.regularizer_weight + 1.0
        
        # Module Versions:
        # loss = PolicyRegularizer.apply(decoder_log_probs, lstm_log_probs).mean() * self.regularizer_weight
        return loss


    def run_epoch_vanilla_vae(self, phase, epoch = None):
        if phase == 'train':
            print(f'Training epoch {epoch}')
            self.model.reparam = True
            self.model.train()
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

            if self.unconditional:
                prop = torch.zeros_like(prop)

            # Teacher forcing with teacher forcing prob probability
            use_tf = (random.random() < self.teacher_forcing_prob)

            recon_loss, kl_loss = self.model.forward(x_inputs=inp, y_inputs=prop, true_binary=tgt, teacher_forcing=use_tf)

            ## From here on out I'm winging it
            batch_loss = recon_loss + kl_loss

            # Regualrizer Loss (Doesn't Use Teacher Forcing) # WARNING: SAMPLING
            if self.regularizer_type == 'KLD':
                regularizer_loss = self.compute_regularizer_loss_kld(props= prop, sample = True)
            # elif self.regularizer_type == 'Pol1':
            #     regularizer_loss = self.compute_regularizer_loss_pol()
            elif self.regularizer_type == 'Pol2':
                # regularizer_loss = self.compute_regularizer_loss_pol_2(props = prop, n_lstm_samples = 5)
                regularizer_loss = self.alternate_policy_2_regularizer(props = prop, n_lstm_samples = 5)
            elif self.regularizer_type == 'Pol':
                # regularizer_loss = self.compute_regularizer_loss_pol(props = prop)
                regularizer_loss = self.alternate_policy_regularizer(props = prop)
            elif self.regularizer_type is None:
                regularizer_loss = torch.tensor(0.0, device = self.device)

            batch_loss += regularizer_loss
            # WARNING : Should backprop seperately? I think this is fine but keep an eye on it

            # We therefore expect batch_loss to be of dim batch_size x 1
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.2)
                self.optimizer.step()

            total_batches += 1
            batch_loss = batch_loss.detach()
            recon_loss = recon_loss.detach()
            kl_loss = kl_loss.detach()
            reg_loss = regularizer_loss.detach()

            total_epoch_kl += kl_loss.detach()
            total_epoch_recon += recon_loss.detach()
            total_epoch_reg += regularizer_loss.detach()

            # For each batch
            if total_batches % 20 == 0:
                if phase == 'train':
                    pbar.set_description(f'Epoch: {epoch} | Recon: {recon_loss} - KL {kl_loss} - REG {reg_loss} | Total {batch_loss}')
                else:
                    pbar.set_description(f'Validation Lossses | Recon: {recon_loss} - KL {kl_loss} - REG {reg_loss} | Total {batch_loss}')
            pbar.update(1)


        pbar.close()
        avg_loss_recon = total_epoch_recon / total_batches
        avg_loss_kl = total_epoch_kl / total_batches
        avg_reg_loss = total_epoch_reg / total_batches

        avg_loss_total = avg_loss_kl + avg_loss_recon + avg_reg_loss

        if phase == 'train':
            print(f' --- Epoch {epoch} Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - AVG REG {avg_reg_loss}| AVG Total {avg_loss_total}')
        else:
            print(f'--- Validation Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - AVG REG {avg_reg_loss}| AVG Total {avg_loss_total}')
        # Report final loss
        # TODO: Return total avg loss in epoch
        return avg_loss_total, avg_loss_recon, avg_loss_kl, avg_reg_loss

    def log_probs_from_logits(self, actions, logits):
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

        return total_log_probs


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

        # Sample from LSTM
        with torch.no_grad():
            start_tokens = torch.ones(
                                (props_ext.shape[0], 1),
                                dtype=torch.int,
                                device = self.model.device
                                )
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

        # decoder_logits = self.model.state_decoder(latent_points, props_ext).transpose(0, 1)
        '''
        Decoder expects: 
                x_inputs -> Expects start token in input actions | 
        '''
        initial_tokens = torch.ones(size=[props_ext.shape[0], 1] ,device=self.device, dtype=torch.int64) # Get's ignored

        decoder_inputs = torch.cat( [initial_tokens, lstm_actions] , dim=-1)

        decoder_logits = self.model.state_decoder(
            z = latent_points,
            y = props_ext,
            x_inputs = decoder_inputs,
            teacher_forcing = True,
            return_logits = True,
            return_both = False)

        decoder_logits = decoder_logits[:,1:,:]
        lstm_actions = lstm_actions[:, :-1] # Prune final LSTM action to equalize lengths

        '''
        Decoder Logits: 80 x 36 x 47 | Come with start token to prune
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



    def compute_regularizer_loss_pol_2(self, props, n_lstm_samples = 50):
        '''
        Compute second version of policy regularizer

        props -> property scores 
        n_lstm_samples -> number of samples per property score to sample from the LSTM
            ! sample a seperate latent point for each of these

        \mathbb{E}_{\tilde{p}(x|y)}\log p_\theta (x|y,z) <-- *-1 since doing grad descent not ascent

        Expectated log prob of decoder sample for sampled lstm molecules 
        '''

        '''
        Expand property scores and sample latent points [different latent for each LSTM sample]
        '''
        exp_props = props.repeat_interleave(n_lstm_samples, dim=0)
        latent_points = latent_points = torch.randn(exp_props.shape[0], 
                                    self.model.latent_dim, 
                                    dtype=torch.float32, 
                                    device=self.device) * self.prior_std # * self.model.eps_std

        '''
        Sample LSTM Actions
        '''
        with torch.no_grad():
            # Sample points from lstm
            hidden = self.reg_model.init_hidden(exp_props.shape[0], device)
            lstm_actions, lstm_logits = self.reg_model.forward(x=None, 
                                                            properties = exp_props, 
                                                            hidden = hidden,
                                                            use_teacher_forcing = False, 
                                                            sampling = True, 
                                                            return_actions = False, 
                                                            return_both = True,
                                                            seq_len = self.model.max_decode_steps + 1) # seq_len
            lstm_actions = lstm_actions.detach()
            lstm_logits = lstm_logits.detach()
            # lstm_actions is 320 x 101 Not including start token
            # lstm_logits is 320 x 101 

        '''
        Sample Decoder logits conditioned on LSTM actions partial sequence
        '''
        # state decoder forward:
        # Accepts x_inputs WITH start token
        # Returns logits/actions WITH start token
        start_tokens = torch.ones((lstm_actions.shape[0], 1), dtype=torch.int, device=self.model.device)

        # Concatenate the ones tensor with the original actions tensor along dim=1
        start_lstm_actions = torch.cat((start_tokens, lstm_actions), dim=1)

        start_decoder_logits = self.model.state_decoder.forward(z=latent_points,
                                                y=exp_props,
                                                x_inputs = lstm_actions, 
                                                teacher_forcing = True,
                                                return_logits = True)

        decoder_logits = start_decoder_logits[:, 1:, :]
        
        # decoder_logits is 320 x 100 x 47

        # Prune last lstm_action to make sequences the same length
        lstm_actions_pruned = lstm_actions[:, :-1]
        # lstm_actions is 320 x 100
        '''
        Compute log probs of lstm actions under decoder logits
        '''

        log_probs = self.log_probs_from_logits(actions = lstm_actions_pruned, logits = decoder_logits)

        # log_probs is 320
        '''
        Weight mean and return
        '''

        return -1 * self.regularizer_weight * log_probs.mean()


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
        # Expand properties and latent by decoder sample size
        expanded_latent = input_latent.repeat_interleave(decoder_sample_size, dim=0)
        expanded_properties = input_props.repeat_interleave(decoder_sample_size, dim=0)

        # Sample actions and logits from decoder
        sampled_actions, sampled_logits = self.model.state_decoder(
                                                    z = expanded_latent, 
                                                    y = expanded_properties,
                                                    x_inputs = None, 
                                                    teacher_forcing = False, 
                                                    return_logits = True,
                                                    return_both = True)

        sampled_actions = sampled_actions[:, 1:]
        sampled_logits = sampled_logits[:, 1:, :]

        # compute probabilities based on the sampled actions
        log_probs = self.log_probs_from_logits(actions= sampled_actions, logits= sampled_logits)

        # log_probs is 256

        return log_probs, sampled_actions, expanded_properties

    def compute_regularizer_loss_pol(self, props, prop_samp_size = None, latent_sample_size = 2, decoder_sample_size = 10, old_version= False): #4, 10
        '''
        props <-- batch_size number of properties
        prop_sample_size <-- number of properties to sample
        latent_sample_size <-- number of latent points to sample per property
        decoder_sample_size <-- number of x to sample for decoder for each latent point
        '''
        '''
        1. Sample latent points
        '''
        # print('Warning: Fix parameter values')

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
        latent_points = torch.randn(properties.shape[0], self.model.latent_dim, device=self.model.device, dtype=torch.float32) * self.prior_std # * self.model.eps_std

        # exp_props is [batch_size x latent_sample_size , 1]
        # exp_latents is [batch_size x latent_sample_size , latent_size]

        '''
        2. Sample Decoder log likelyhoods and decoder actions
        '''
        
        decoder_log_probs, decoder_actions, expanded_properties = self._sample_decoder_actions_and_likelihoods(
                                                input_latent = latent_points, 
                                                input_props = properties, 
                                                decoder_sample_size=decoder_sample_size)

        '''
        3. Compute log likelyhoods of decoder actions under lstm logits
        '''

        with torch.no_grad():

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
            
            # decoder_actions is 256 x 100
            # lstm_logits is 256 x 100 x 47
            lstm_log_probs = self.log_probs_from_logits(actions=decoder_actions, logits=lstm_logits)

        '''
        4. Compute forward pass of regularizer
        '''
        if old_version:
            preg_loss = (-1) * ((lstm_log_probs) * decoder_log_probs ).mean()
        else:
            preg_loss = PolicyRegularizer.apply(decoder_log_probs, lstm_log_probs).mean()

        '''
        5. Average out to get final loss, weight and return
        '''

        return self.regularizer_weight * preg_loss

        

if __name__ == "__main__":
        '''
        RUN PARAMETERS
        '''

        dset = 'QM9_36'
        device = 'cpu'
        regularizer_type = 'Pol' # 'KLD' # 'Pol2' # in None, 'KLD', 'Pol', 'Pol2'
        
        teacher_forcing_prob = 0.0
        run_save_folder = '../../GB-QM9-PRIOR/'

        prior_std = 1.0 # 0.03939  # prior STD for regularizer sampling
        batch_size = 128 # 16 # 128

        beta = 0.001
        eps_std = 1.0


        if 'ZINC' in dset:
            latent_size = 156
        if 'QM9' in dset:
            latent_size = 192

        '''
        BETA ANNEALING
        '''
        beta_anneal_funct = None

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

        learning_rate  = 1e-3

        if regularizer_type == 'KLD':
            reg_weight = 0.1
        else:
            reg_weight = 0.01


        run_name = 'EXP-cVAE'
        if dset == 'ZINC120':
            data_path = '../data/ZINC120/'
            max_seq_len = 121
            run_name += 'ZINC120'
        elif dset == 'ZINC':
            data_path = '../data/ZINC250K/'
            max_seq_len = 111
            run_name += '-ZINC'
        elif dset == 'QM9':
            data_path = '../data/QM9/'
            max_seq_len = 101
            run_name += '-QM9'
        elif dset == 'QM9_36':
            data_path = '../data/QM9_36/'
            max_seq_len = 37
            run_name += 'QM9_36'

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_y, valid_x_enc, valid_y = load_smiles_and_properties(data_path)

        if dset == 'QM9_36' and regularizer_type is not None:
             if teacher_forcing_prob == 1.0:
                  # prior_model_path = './pretrained_models/SD_LSTM_odd-sunset-14_Epoch_9_Vl_0.589'
                  prior_model_path = './pretrained_models/EXP-cVAE-QM9_36-TF_Vl_0.311'
             else:
                  prior_model_path = './pretrained_models/EXP-cVAE-QM9_36-E50-Vl_0.481'
        elif dset == 'ZINC120' and regularizer_type is not None:
             if teacher_forcing_prob == 1.0:
                 prior_model_path = './pretrained_models/TF-cVAE-ZINC-Vl0481'
             else:
                 # prior_model_path = './pretrained_models/EXP-cVAE-ZINC120-Vl0675'
                 print('REMOVE THIS')
                 prior_model_path = './pretrained_models/EXP-KLD-ZINC-Epoch198'


        if regularizer_type is not None:
            '''
            # Pretrained Model
            prior_model_path = '../../GB-ZINC/EXP-cVAEZINC120-KLD/SD_REG_VANILLA_VAE_tight-boat-23_Epoch_237_Vl_0.249'
            # beta_anneal_funct = None
            # beta = 1.0

            prior_model = load_model(model_class=VanillaMolVAE, 
                                    model_definition=prior_model_path + '.json',
                                    model_weights=prior_model_path + '.pt',
                                    device=device)

            prior_model = prior_model.to(device)
            prior_model.device = device
            prior_model.encoder.device = device
            prior_model.state_decoder.device = device
            print('####################')
            print(prior_model)
            print('####################')
            '''
            prior_model = None
            run_name += '-' + regularizer_type
        else:
            reg_weight = 0.0
            prior_model = None

        if teacher_forcing_prob != 0.0:
            run_name += ('-TF' + str(teacher_forcing_prob).replace('.',''))

        if 'QM9' in dset:
            reg_model_params = './reg_models/LSTM_12_1.180'
        elif 'ZINC' in dset:
            reg_model_params = './reg_models/LSTM_ZINC120_46_0.461'

        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = (run_save_folder + run_name)

        # Instantiate Trainer Class
        trainer =  RegularizedVAETrainer(
                regularizer_type = regularizer_type,
                regularizer_model_path = reg_model_params,
                regularizer_weight = reg_weight,

                train_x_enc=train_x_enc,
                train_y=train_y,
                valid_x_enc=valid_x_enc,
                valid_y=valid_y,
                property_names=["LogP"],
                device=device,
                batch_size=batch_size,
                latent_dim=latent_size,
                beta=beta,
                max_seq_len=max_seq_len,
                eps_std=eps_std,
                model=prior_model,
                decoder_embedding_dim=47,
                learning_rate = learning_rate,
                model_save_dir = (run_save_folder + run_name),
                save_every_n_val_cycles = 1,
                max_epochs = 500,
                teacher_forcing_prob = teacher_forcing_prob,
                wandb_project = run_name,
                early_stopping = 50,
                prior_std=prior_std,
                beta_annealing_function = beta_anneal_funct,
                )

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, valid_x_enc, train_y, valid_y

        trainer.fit()
