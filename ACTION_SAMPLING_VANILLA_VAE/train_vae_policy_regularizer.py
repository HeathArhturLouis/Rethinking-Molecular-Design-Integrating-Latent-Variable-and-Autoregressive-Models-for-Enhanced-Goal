import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

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

N_WORKERS_DL = 4

class PolicyRegVAETrainer:
    def __init__(self,
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
                 valid_every_n_epochs = 1, save_every_n_val_cycles = 3, max_epochs = 100,
                 teacher_forcing_prob = 0.1,
                 early_stopping = 10):

        # Number of epochs without val loss improvement before training terminates
        self.early_stopping = early_stopping

        # Regularizer Training Params
        self.teacher_forcing_prob = teacher_forcing_prob
        self.regularizer_param_path = regularizer_param_path
        self.regularizer_weight = regularizer_weight


        # Load Regularizer model
        self.reg_model = load_model(model_class=ConditionalSmilesRnn, 
                                    model_definition=regularizer_param_path, 
                                    model_weights=regularizer_model_path, 
                                    device=device) # Should be on device
        
        self.reg_model = self.reg_model.eval()


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
        
        if model == None:
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

        self.valid_every_n_epochs = valid_every_n_epochs
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

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, min_lr=0.000001)

        self.sd = SmilesCharDictionary()


    def _compute_mol_log_probs_under_lstm(self, actions, props):
        '''
        Returns the log probability of a string of actions under the LSTM:
         - Up untill (inclusive) the first termination characer in the action sequence
         - Does not track gradients

        Return: log_prob; tensor of shape b_size (size of actions / props) 
        '''
        actions = actions.clone()
        props = props.clone()

        assert props.shape[0] == actions.shape[0]

        with torch.no_grad():
            # Do LSTM stuff inside loop, since batching alredy tied to decoder_sample_size
            # Compute nll under LSTM

            # decoder_actions is 4 x 101 | decoder_sample_size x max_seq_len

            # Get logits for sequences
            hidden = self.reg_model.init_hidden(actions.shape[0], self.model.device)

            # LSTM expects start tokens in actions
            lstm_logits = self.reg_model.forward(x=actions,
                                                properties=props,
                                                hidden = hidden,
                                                use_teacher_forcing = True,
                                                sampling = True,
                                                return_actions = False,
                                                return_both = False,
                                                seq_len = actions.shape[1])[0]

            # lstm_logits is: decoder_sample_size x decoder_action.shape[1] x decision_dim
            # is: 4 x 100 x 47

            # To get the correct action at this step we shift decoder actions forward by one
            actions = actions[:, 1:]
            # decoder_actions is now: decoder_sample_size x (seq_len - 1) and contains the actions suceeding the start token
            # decoder_actions is 4 x 100

            # We need to get the probabilities, take their log and sum the one's corresponding to decoder_actions

            # subtract max for numerical stability
            max_logit = torch.max(lstm_logits, dim=-1, keepdim=True)[0]
            lstm_probs = F.softmax( lstm_logits - max_logit , dim=-1)
            lstm_log_probs = torch.log(lstm_probs)


            # lstm_probs is shape decoder_sample_size x seq_len-1 x decision_dim
            # lstm_probs is 4 x 100 x 47 and contains probabilities along last dimension 


            # Gather probabilities corresponding to taken actions

            # lstm_probs_selected = torch.gather(lstm_probs, 2, actions.unsqueeze(2)).squeeze(2)
            lstm_logp_selected = torch.gather(lstm_log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)

            # lstm_probs is: decoder_sample_size x seq_len-1
            # lstm_probs is: 4 x 100

            # Mask elements after first occurace of end token in decoder_actions
            # NOTE: Since we're summing mask to 0, else this should change
                
            mask = (actions == self.sd.char_idx[self.sd.END])

            # Cumulative mask to identify elements (strictly) after the first occurrence of 2
            cumulative_mask = mask.cumsum(dim=1) > 1 # each element of cum sum is equal to the number of previous occurances, so > 1 is one past the first occurace

            # Modify lstm_probs based on the mask
            lstm_logp_selected[cumulative_mask] = 0

            # Sum along sequence length dimension and take the negative
            lps = lstm_logp_selected.sum(dim = -1)
            # nll is [decoder_sample_size]

            return lps

    def _compute_mol_log_probs_under_decoder(self, decoder_actions, decoder_logits):
        '''
        Returns log probability of string generated by decoder by summing locally normalized probabilities

        ! No slicing / detachment of decoder_logits

        decoder_actions : decoder_sample_size x max_seq_len
         - 4 x 101
         - contains start token 
        decoder_logits : decoder_sample_size x max_seq_len x decision_dim
         - 4 x 101 x 47
         - contains start token probability 1

        return: log_probs : 
        '''
        # Remove start tokens
        decoder_actions = decoder_actions[:, 1:]
        decoder_logits = decoder_logits[:, 1:, :]

        # Convert logits to log probabilities | subtract max logit for numerical stability
        max_logit = torch.max(decoder_logits, dim=-1, keepdim=True)[0]
        decoder_probs = F.softmax(decoder_logits - torch.max(max_logit) , dim=-1)
        decoder_logp = torch.log(decoder_probs)

        # Select log probs based on actions to get the sequences of loc normd probs
        decoder_logp_selected = torch.gather(decoder_logp, 2, decoder_actions.unsqueeze(-1)).squeeze(-1)

        # Mask probabilities after first termination action
        mask = (decoder_actions == self.sd.char_idx[self.sd.END])

        # Cumulative mask to identify elements (strictly) after the first occurrence of 2
        cumulative_mask = mask.cumsum(dim=1) > 1 # each element of cum sum is equal to the number of previous occurances, so > 1 is one past the first occurace

        # Modify lstm_probs based on the mask
        decoder_logp_selected[cumulative_mask] = 0

        # Sum up and return
        return decoder_logp_selected.sum(dim = -1)

    def compute_policy_regularizer(self, props, optimizer, prop_samp_size = 4, latent_sample_size = 2, decoder_sample_size = 4):
        '''
        props <-- batch_size number of properties
        prop_sample_size <-- number of properties to sample
        latent_sample_size <-- number of latent points to sample per property
        decoder_sample_size <-- number of 

        TODO: Find way to get more properties if needed, by default I'll fix the number of properties to sample to the batch size
        TODO: Amina mentioned I need to write the bw pass for this, perhaps my way doesn't work after all
        '''

        # 1. Compute sample of points from decoder
        #       a. sample latent points, latent_sample_size for each y
        assert prop_samp_size <= props.shape[0]

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
        decoder_actions = torch.zeros((n_latent_prop_pairs * decoder_sample_size, self.model.max_decode_steps) ,dtype=torch.int64, device=self.model.device)

        total_sample_size = n_latent_prop_pairs * decoder_sample_size
        # torch.Size([640, 101])

        # decoder_actions should be [total samples of from decoder] x seq_length = (p_size * z_per_p * x_per_zp_pair) x max_decode_steps
        # = (n_latent_prop_pairs * decoder_sample_size) x max_decode_steps = 128 x 101
        # (64 * 2 * 5) x 101 = 640 x 101
        
        # we now sample from the decoder decoder_sample_size times for each latent-property pair
        all_decoder_actions = torch.zeros((total_sample_size, self.model.max_decode_steps) ,dtype=torch.int64)  

        # LSTM negative log likelyhoods -- what we want to minimize
        lstm_log_probs = torch.zeros( [total_sample_size], dtype=torch.float32, device=self.model.device)
        decoder_log_probs = torch.zeros( [total_sample_size], dtype=torch.float32, device=self.model.device)
        

        #       b. decode a molecule (actions)
        for dec_input_ind in range(n_latent_prop_pairs):
            input_props = properties[dec_input_ind].unsqueeze(0).repeat_interleave(decoder_sample_size, dim=0)
            input_latent = latent_points[dec_input_ind, :].unsqueeze(0).repeat_interleave(decoder_sample_size, dim=0)

            # input props is same propr decoder_sample_size times = 5 x 1
            # input latent is latent point decoder_sample_size times = 5 x 56
            # likelyhoods = torch.zeros([decoder_sample_size], dtype=torch.float32, device=self.model.device)

            # 
            decoder_actions, decoder_logits = self.model.state_decoder.forward(
                                                z=input_latent,
                                                y=input_props,
                                                x_inputs = None,
                                                teacher_forcing = False,
                                                return_logits = False,
                                                return_both = True
                                                )

            # Logits has requires grad set to true, actions does not

            all_decoder_actions[dec_input_ind * decoder_sample_size : (dec_input_ind + 1) * decoder_sample_size, :] = decoder_actions
            # Decoder actions contains start token

            '''
            COMPUTE LOG PROBABILITIES (SUM OF PARTIAL) OF DECODER ACTIONS
            '''

            reg_log_probs = self._compute_mol_log_probs_under_lstm(props = input_props, actions = decoder_actions)

            # Store nll weights
            lstm_log_probs[dec_input_ind * decoder_sample_size : (dec_input_ind + 1) * decoder_sample_size] = reg_log_probs

            '''
            COMPUTE LOG PROBABILITIES (SUM OF PARTIAL) OF DECODER GENERAITONS UNDER DECODER
            retain graph for these since we're calling backwards later to get gradients
            '''

            # TODO: Is it worth moving this into decoder code?
            dec_log_probs = self._compute_mol_log_probs_under_decoder(decoder_actions, decoder_logits)
            decoder_log_probs[dec_input_ind * decoder_sample_size : (dec_input_ind + 1) * decoder_sample_size] = dec_log_probs

        '''
        Compute (negative) Expectation over all sampled points and decodings of:

          - log(\tilde{p}(x|y_i))\nabla_\theta p_\theta(x | y_i, z_j) =
        = - lstm_lp * grad( decoder_lp )

        Update decoder weights (LSTM  fixed) to maximize the product of log probs of lstm times the gradients wrt. parameters of logprobs of decoder
        '''


        ''' THIS IS PROBABLY CORRECT BUT TERRIBLY SLOW '''

        # decoder_log_probs is [32]
        # we want the negative weights
        lstm_log_probs *= -1

        # Weight the gradient weights by the regularizer scaling hyperparameter
        lstm_log_probs *= self.regularizer_weight

        # Scale for sample size
        lstm_log_probs /= total_sample_size

        # decoder_grads is shape 
        # decoder_grads[0] is 47 x 56
        # decoder_grads[1] is 1503 x 113
        # lstm_log_probs and decoder_log_probs are both [32]

        accumulated_grads = {name: torch.zeros_like(param) for name, param in self.model.state_decoder.named_parameters()}

        # Compute gradients for each element in decoder_log_probs
        for idx, prob in enumerate(decoder_log_probs):
            # Compute gradients for the current log probability element
            current_grads = torch.autograd.grad(
                outputs=prob,
                inputs=self.model.state_decoder.parameters(),
                create_graph=True,
                retain_graph=True if idx < len(decoder_log_probs) - 1 else False  # Retain graph only if not the last prob
            )

            # Weight current gradients w/ lstm_log_prob and accumulate
            for (name, param), grad in zip(self.model.state_decoder.named_parameters(), current_grads):
                accumulated_grads[name] += grad * lstm_log_probs[idx]

        # Apply the accumulated, weighted gradients
        with torch.no_grad():
            for name, param in self.model.state_decoder.named_parameters():
                param.grad = accumulated_grads[name]
        
        # Step the optimizer
        optimizer.step()
        optimizer.zero_grad()



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
            'valid_every_n_epochs': self.valid_every_n_epochs,
            'save_every_n_val_cycles': self.save_every_n_val_cycles,
            'teacher_forcing_prob': self.teacher_forcing_prob,
            'regularizer_param_path':self.regularizer_param_path,
            'regularizer_weight':self.regularizer_weight
        }
        
        # Write the parameters dictionary to a JSON file
        with open(params_file_path, 'w') as f:
            json.dump(params, f, indent=4)
        
        print(f"Training parameters saved to {params_file_path}")

    def fit(self):
        '''
        Complete training run of model
        '''
        best_val_epoch = 0

        # TODO: Early stopping 
        haikunator = Haikunator()
        run_id = haikunator.haikunate(token_length=2)
        # Save run parameters
        self.save_params(run_name=run_id)

        print(f'Training run ID: {run_id}')

        best_val_loss = np.inf
        # Save training parameters
        # TODO: Implement
        for epoch_no in range(1, self.max_epochs + 1):
            # Train for epoch
            torch.cuda.empty_cache()
            avg_epoch_loss = self.run_epoch_vanilla_vae('train', epoch=epoch_no)

            # if val every n epochs
            if epoch_no % self.valid_every_n_epochs == 0:
                # Run validation
                with torch.no_grad():
                    avg_valid_loss = self.run_epoch_vanilla_vae('valid', epoch=epoch_no)

                    self.lr_scheduler.step(avg_valid_loss)
                # If val better or save every n validations
                if avg_valid_loss <= best_val_loss or epoch_no % save_every_n_val_cycles == 0:
                    # Save model
                    # TODO: Implement
                    # Report
                    print(f'Achieved Best Valid Loss so Far: {avg_valid_loss}')
                    self._save_model( self.model_save_dir, run_id, epoch_no, avg_valid_loss)
        
            if avg_valid_loss < best_val_loss:
                best_val_epoch = epoch_no
            else:
                if epoch_no - best_val_epoch > self.early_stopping:
                    # Early termination, save last model
                    print()
                    print(f'EARLY STOPPING: {best_val_epoch} was best epoch.')
                    print()                    
                    return


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

            # Teacher forcing with teacher forcing prob probability
            use_tf = (random.random() < self.teacher_forcing_prob)

            recon_loss, kl_loss = self.model.forward(x_inputs=inp, y_inputs=prop, true_binary=tgt, teacher_forcing=use_tf)

            ## From here on out I'm winging it
            batch_loss = recon_loss + kl_loss

            # WARNING : Should backprop seperately? I think this is fine but keep an eye on it

            # We therefore expect batch_loss to be of dim batch_size x 1
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), 0.2)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.compute_policy_regularizer(props= prop, optimizer=self.optimizer) # redundant zero grad?

            total_batches += 1
            batch_loss = batch_loss.detach()
            recon_loss = recon_loss.detach()
            kl_loss = kl_loss.detach()

            total_epoch_kl += kl_loss.detach()
            total_epoch_recon += recon_loss.detach()


            # For each batch
            if total_batches % 20 == 0:
                if phase == 'train':
                    pbar.set_description(f'Epoch: {epoch} | Recon: {recon_loss} - KL {kl_loss} | Total {batch_loss}')
                else:
                    pbar.set_description(f'Validation Lossses | Recon: {recon_loss} - KL {kl_loss} | Total {batch_loss}')
            pbar.update(1)


        pbar.close()
        avg_loss_recon = total_epoch_recon / total_batches
        avg_loss_kl = total_epoch_kl / total_batches

        avg_loss_total = avg_loss_kl + avg_loss_recon

        if phase == 'train':
            print(f' --- Epoch {epoch} Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} | AVG Total {avg_loss_total}')
        else:
            print(f'--- Validation Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} | AVG Total {avg_loss_total}')
        # Report final loss
        # TODO: Return total avg loss in epoch
        return avg_loss_total


if __name__ == "__main__":
        data_path = '../data/QM9/'

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_y, valid_x_enc, valid_y = load_smiles_and_properties(data_path)

        #Regularizer Model
        reg_model_params = './reg_models/LSTM_12_1.180.json'
        reg_model_weights = './reg_models/LSTM_12_1.180.pt'

        prior_model_path = './pretrained_models/SD_LSTM_odd-sunset-14_Epoch_9_Vl_0.589'

        reg_weight = 0.01


        device = 'cpu'

        # Pretrained Model
        prior_model = load_model(model_class=VanillaMolVAE, 
                                model_definition=prior_model_path + '.json',
                                model_weights=prior_model_path + '.pt',
                                device=device)
        
        prior_model = prior_model.to('cpu')
        prior_model.device = 'cpu'
        prior_model.encoder.device = 'cpu'
        prior_model.state_decoder.device = 'cpu'

        # Instantiate Trainer Class
        trainer =  PolicyRegVAETrainer(
                regularizer_model_path = reg_model_weights,
                regularizer_param_path = reg_model_params,
                regularizer_weight = reg_weight,
                train_x_enc=train_x_enc,
                train_y=train_y,
                valid_x_enc=valid_x_enc,
                valid_y=valid_y,
                property_names=["LogP"],
                device=device,
                batch_size=64,
                latent_dim=56,
                beta=1.0,
                max_seq_len=101,
                eps_std=0.01,
                model=prior_model,
                decoder_embedding_dim=47,
                learning_rate = 1e-4,
                model_save_dir = '../models/VANILLA_VAE_QM9_3_Layer/', 
                valid_every_n_epochs = 1, 
                save_every_n_val_cycles = 1,
                max_epochs = 500,
                teacher_forcing_prob = 0.1
                )

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, valid_x_enc, train_y, valid_y

        trainer.fit()