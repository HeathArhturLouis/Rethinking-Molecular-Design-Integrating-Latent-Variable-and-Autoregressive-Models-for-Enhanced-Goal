
from sd_lstm_model import ConditionalSDLSTM
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import sys

import os

import numpy as np
from tqdm import tqdm

sys.path.append('../utils/')
from sd_lstm_utils import load_encodings_masks_and_properties, get_tensor_dataset, get_tensor_dataset_alt, save_model
from sd_loss_computation import my_perp_loss, my_binary_loss, PerpCalculator, masked_ce


from haikunator import Haikunator
import json

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn.utils import clip_grad_norm_


class SD_LSTM_Trainer:
    def __init__(self, train_x_enc, train_x_masks, train_y, valid_x_enc, valid_x_masks, valid_y, property_names,
                 model=None, device = 'cpu', batch_size = None, 
                 hidden_size = 512, n_layers = 3, rnn_dropout = 0.2, learning_rate = 1e-3,
                 rules_dict_size = 80, max_seq_len = 100, 
                 model_save_dir = '../models/SD_LSTM_FR/', 
                 valid_every_n_epochs = 1, save_every_n_val_cycles = 3, max_epochs = 100):
        
        all_y = np.concatenate((train_y, valid_y), axis=0)

        mean = np.mean(all_y, axis = 0)
        std = np.std(all_y, axis = 0)

        n_props = train_y.shape[-1] # Should be 1

        if model == None:
            self.model = ConditionalSDLSTM(input_size=rules_dict_size,
                property_size=n_props,
                property_names=property_names,
                pnorm_means=mean,
                pnorm_stds=std,
                hidden_size=hidden_size,
                output_size=rules_dict_size,
                n_layers=n_layers,
                rnn_dropout=rnn_dropout,
                max_rules=max_rules,
                rules_dict_size=rules_dict_size
                )
        else:
            self.model = model.to(device)

        train_y = self.model.normalize_prop_scores(train_y)
        valid_y = self.model.normalize_prop_scores(valid_y)
        
        train_set = get_tensor_dataset(train_x_enc, train_x_masks, train_y)
        valid_set = get_tensor_dataset(valid_x_enc, valid_x_masks, valid_y)

        self.train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

        # TODO: MAke sure this works
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.learning_rate = learning_rate

        self.valid_every_n_epochs = valid_every_n_epochs
        self.save_every_n_val_cycles = save_every_n_val_cycles

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.loss_fnct = masked_ce

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

    def _save_model(self, base_dir, info, epoch_no ,valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'SD_LSTM_{info}_Epoch_{epoch_no}_Vl_{valid_loss:.3f}'
        print(base_name)
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
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'rnn_dropout': self.rnn_dropout,
            'learning_rate': self.learning_rate,
            'max_epochs': self.max_epochs,
            'rules_dict_size': self.model.input_size,  # Assuming input_size is accessible in your model
            'max_seq_len': self.model.max_rules,  # Assuming max_rules corresponds to max sequence length in your model
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

        self.save_params(run_name=run_id)

        print(f'Training run ID: {run_id}')

        best_val_loss = np.inf
        # Save training parameters
        # TODO: Implement
        for epoch_no in range(1, self.max_epochs + 1):
            # Train for epoch
            torch.cuda.empty_cache()
            avg_epoch_loss = self.run_epoch_sd_lstm('train', epoch=epoch_no)

            # if val every n epochs
            if epoch_no % self.valid_every_n_epochs == 0:
                # Run validation
                with torch.no_grad():
                    avg_valid_loss = self.run_epoch_sd_lstm('valid', epoch=epoch_no)

                    # self.lr_scheduler.step(avg_valid_loss)
                # If val better or save every n validations
                if avg_valid_loss <= best_val_loss or epoch_no % save_every_n_val_cycles == 0:
                    # Save model
                    # TODO: Implement
                    # Report
                    print(f'Achieved Best Valid Loss so Far: {avg_valid_loss}')
                    self._save_model( self.model_save_dir, run_id, epoch_no, avg_valid_loss)


    def run_epoch_sd_lstm(self, phase, epoch = None):
        # TODO: Typecast inpts during data proc
        # TODO: Handle Device
        # TODO: Add zeros to end instead of 0...1s

        if phase == 'train':
            print(f'Training epoch {epoch}')
            self.model.train()
            data_loader = self.train_data_loader
        else:
            print('Running Validation:')
            self.model.eval()
            data_loader = self.val_data_loader


        # TODO: Progress bar
        n_batches = len(data_loader.dataset) // data_loader.batch_size + (len(data_loader.dataset) % data_loader.batch_size != 0)
        pbar = tqdm(range(0, n_batches), unit='batch')

        total_epoch_loss = 0
        total_batches = 0

        for bindex, (inp, tgt, mask, prop) in enumerate(data_loader):
            # Send everything to device
            inp, tgt, mask, prop = inp.to(self.device), tgt.to(self.device), mask.to(self.device), prop.to(self.device)

            hidden = self.model.init_hidden(bsz=inp.size(0), device=self.device)
            output_logits, new_hidden = self.model(inp, prop, hidden)
            # Outputs are prediction for entire batch of size
            # inps are shape batch x max_seq_len x decision_dim
            # Masks are shape  batch x (max_seq_len - 1) x decision_dim
            # output is batch x (max_seq_len -1) x decision_dim
            # tgt is shifted by 1 forwards inputs: batch x (max_seq_len - 1) x decision_dim 
            
            # Compute loss : TODO: Reorder dimensions? 

            batch_loss = self.loss_fnct(tgt, mask, output_logits)[0] # This is not neccesary/ data_loader.batch_size

            # We therefore expect batch_loss to be of dim batch_size x 1
            if phase == 'train':
                self.optimizer.zero_grad()
                batch_loss.backward()

                self.optimizer.step()

            # Detach batch loss to save memory
            batch_loss = batch_loss.detach().item()
            # Update total epoch loss
            # Update total_batches 
            total_batches += 1
            total_epoch_loss += batch_loss

            # For each batch
            if total_batches % 20 == 0:
                if phase == 'train':
                    pbar.set_description(f'Epoch: {epoch} | Train batch loss: { batch_loss }')
                else:
                    pbar.set_description(f'Validation batch loss: {batch_loss}')
            pbar.update(1)


        pbar.close()
        avg_loss = total_epoch_loss / total_batches
        if phase == 'train':
            print(f' --- Epoch {epoch} Avg Loss: {avg_loss}')
        else:
            print(f'--- Validation Avg Loss: {avg_loss}')
        # Report final loss

        # TODO: Return
        return avg_loss


if __name__ == "__main__":
        data_path = '../data/QM9/'
        rules_dict_size = 80
        max_rules = 100
        property_size = 1
        property_names = ['LogP']
        hidden_size = 512
        rules_dict_size = rules_dict_size
        n_layers = 3
        rnn_dropout = 0.2
        batch_size = 64
        learning_rate = 1e-3

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_x_masks, train_y, valid_x_enc, valid_x_masks, valid_y = load_encodings_masks_and_properties(data_path)

        trainer = SD_LSTM_Trainer(train_x_enc=train_x_enc,
                                train_x_masks=train_x_masks,
                                train_y=train_y,
                                valid_x_enc=valid_x_enc,
                                valid_x_masks=valid_x_masks,
                                valid_y=valid_y,
                                property_names=property_names,
                                model_save_dir='../models/SD_LSTM_QM9_MASKED_CROSS_ENTROPY_TOKENS',
                                n_layers=n_layers,
                                batch_size=batch_size)

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, train_x_masks, valid_x_enc, valid_x_masks, train_y, valid_y

        trainer.fit()