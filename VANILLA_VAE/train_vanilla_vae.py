
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import sys

import os

import numpy as np
from tqdm import tqdm

sys.path.append('../utils/')
from rnn_utils import load_smiles_and_properties, get_tensor_dataset, save_model
from smiles_char_dict import SmilesCharDictionary


from haikunator import Haikunator
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_


from model_vanilla_vae import VanillaMolVAE


class VanillaVAETrainer:
    def __init__(self, train_x_enc, train_y, valid_x_enc, valid_y, property_names,
                 model, device, batch_size,
                 latent_dim=56,
                 beta=1.0,
                 max_seq_len=100,
                 eps_std=0.01,
                 decoder_embedding_dim=47,# Prev was tok size
                 learning_rate = 1e-3,  model_save_dir = '../models/VANILLA_VAE/', 
                 valid_every_n_epochs = 1, save_every_n_val_cycles = 3, max_epochs = 100):        

        # Not implemented yet
        assert model is None
        
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

        self.train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        self.valid_every_n_epochs = valid_every_n_epochs
        self.save_every_n_val_cycles = save_every_n_val_cycles

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.max_epochs = max_epochs

        self.train_losses = []
        self.val_losses = []

        self.model_save_dir = model_save_dir

        if batch_size is None:
            self.batch_size = train_data_loader.batch_size
        else:
            self.batch_size = batch_size

        os.makedirs(self.model_save_dir, exist_ok=True)

        # self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, min_lr=0.001)

    def _save_model(self, base_dir, info, epoch_no, valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'SD_LSTM_{info}_Epoch_{epoch_no}_Vl_{valid_loss:.3f}'
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
            'save_every_n_val_cycles': self.save_every_n_val_cycles
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

                    # self.lr_scheduler.step(avg_valid_loss)
                # If val better or save every n validations
                if avg_valid_loss <= best_val_loss or epoch_no % save_every_n_val_cycles == 0:
                    # Save model
                    # TODO: Implement
                    # Report
                    print(f'Achieved Best Valid Loss so Far: {avg_valid_loss}')
                    self._save_model( self.model_save_dir, run_id, epoch_no, avg_valid_loss)


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

        for bindex, (inp, tgt, prop) in enumerate(data_loader):
            # Send everything to device
            inp, tgt, prop = inp.to(self.device), tgt.to(self.device), prop.to(self.device)

            # inp.shape - batch_size x max_seq_len
            # tgt.shape - batch_size x max_seq_len
            # prop.shape - batch_size x 1 ??
            recon_loss, kl_loss = self.model.forward(x_inputs=inp, y_inputs=prop, true_binary=tgt)


            ## From here on out I'm winging it
            batch_loss = recon_loss + kl_loss

            # We therefore expect batch_loss to be of dim batch_size x 1
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            total_batches += 1
            batch_loss = batch_loss.detach()
            recon_loss = recon_loss.detach()
            kl_loss = kl_loss.detach()

            total_epoch_kl += kl_loss.detach()
            total_epoch_recon += recon_loss.detach()


            # For each batch
            if total_batches % 20 == 0:
                if phase == 'train':
                    pbar.set_description(f'Epoch: {epoch} | Recon: {recon_loss} - KL {kl_loss} - Total {batch_loss}')
                else:
                    pbar.set_description(f'Validation Lossses | Recon: {recon_loss} - KL {kl_loss} - Total {batch_loss}')
            pbar.update(1)


        pbar.close()
        avg_loss_recon = total_epoch_recon / total_batches
        avg_loss_kl = total_epoch_kl / total_batches
        avg_loss_total = avg_loss_kl + avg_loss_recon

        if phase == 'train':
            print(f' --- Epoch {epoch} Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - AVG Total {avg_loss_total}')
        else:
            print(f'--- Validation Avg Loss | AVG Recon: {avg_loss_recon} - AVG KL {avg_loss_kl} - AVG Total {avg_loss_total}')
        # Report final loss
        # TODO: Return total avg loss in epoch
        return avg_loss_total


if __name__ == "__main__":
        data_path = '../data/QM9/'

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_y, valid_x_enc, valid_y = load_smiles_and_properties(data_path)

        # Instantiate Trainer Class
        trainer =  VanillaVAETrainer(
                train_x_enc=train_x_enc,
                train_y=train_y,
                valid_x_enc=valid_x_enc,
                valid_y=valid_y,
                property_names=["LogP"],
                device='cpu',
                batch_size=64,
                # TODO: LOUIS: This is a bit sus, it's 56 in the other one
                latent_dim=56,
                beta=1.0,
                max_seq_len=101,
                eps_std=0.01,
                model=None,
                decoder_embedding_dim=47,
                learning_rate = 1e-3,
                model_save_dir = '../models/VANILLA_VAE_QM9/', 
                valid_every_n_epochs = 1, 
                save_every_n_val_cycles = 3, 
                max_epochs = 100
                )

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, valid_x_enc, train_y, valid_y
        
        trainer.fit()