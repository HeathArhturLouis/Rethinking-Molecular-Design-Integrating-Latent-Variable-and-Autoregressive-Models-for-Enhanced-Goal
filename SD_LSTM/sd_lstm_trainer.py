import logging
import os
from glob import glob
from time import time
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rdkit import RDLogger

import sys

sys.path.append('../utils/')
from sd_lstm_utils import load_encodings_masks_and_properties, get_tensor_dataset


from sd_smiles_decoder import raw_logit_to_smiles

from property_calculator import my_perp_loss  
from sd_loss_computation import PerpCalculator

from sd_lstm_model import ConditionalSDLSTM

from sd_lstm_utils import save_model, time_since
from rdkit import Chem

import pandas as pd

import torch.autograd.profiler as profiler



class SDLSTMDistributionLearner:
    def __init__(self, data_set, output_dir, n_epochs, hidden_size=512, n_layers=3,
                 max_len=100, batch_size=64, rnn_dropout=0.2, lr=1e-3, valid_every=100, print_every=100, prop_model=None, num_data_workers=0) -> None:
        self.data_set = data_set
        self.n_epochs = n_epochs
        self.output_dir = output_dir
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_len = max_len
        self.batch_size = batch_size
        self.rnn_dropout = rnn_dropout
        self.lr = lr
        self.valid_every = valid_every
        self.print_every = print_every
        self.prop_model = prop_model
        self.seed = 42
        self.num_data_workers = num_data_workers

    def train(self, data_path):
        # Use GPU if available
        cuda_available = torch.cuda.is_available()
        device_str = 'cuda' if cuda_available else 'cpu'
        device = torch.device(device_str)
        print(f'CUDA enabled:\t{cuda_available}')


        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_x_masks, train_y, valid_x_enc, valid_x_masks, valid_y = load_encodings_masks_and_properties(data_path)

        n_props = len(property_names)

        # Normalize property values
        #LOUIS: TODO: Is this neccesary? Since some of the properties have already been normalized?
        #scale the property to fall between -1 and 1
        all_y = np.concatenate((train_y, valid_y), axis=0)
        mean = np.mean(all_y, axis = 0)
        std = np.std(all_y, axis = 0)
        #np.save(data_path + '/normalizer.py', [mean, std])
        train_y = (train_y - mean) / std
        valid_y = (valid_y - mean) / std
        
        # convert to torch tensor, input, output smiles and properties
        # Generate Tensor Datasets
        train_set = get_tensor_dataset(train_x_enc, train_x_masks, train_y)
        valid_set = get_tensor_dataset(valid_x_enc, valid_x_masks, valid_y)

        max_rules = train_x_enc.shape[1] # def 100
        rules_dict_size = train_x_enc.shape[2] # def 80

        # build network, input_size should equal output_size should equal the vocab length

        cond_lstm_model = ConditionalSDLSTM(input_size=rules_dict_size,
                                        property_size=n_props, #PROPERTY_SIZE,
                                        property_names=property_names,# Record names of properties for later
                                        hidden_size=self.hidden_size,
                                        output_size=rules_dict_size,
                                        n_layers=self.n_layers,
                                        rnn_dropout=self.rnn_dropout,
                                        max_rules=max_rules, # Record data params for later
                                        rules_dict_size=rules_dict_size)

        # wire network for training
        optimizer = torch.optim.Adam(cond_lstm_model.parameters(), lr=self.lr)
        
        # Determine loss function

        trainer = SDLSTMTrainer(normalizer_mean = mean,
                                   normalizer_std = std,
                                   model=cond_lstm_model,
                                   optimizer=optimizer,
                                   device=device,
                                   prop_names = property_names,
                                   log_dir=self.output_dir)


        trainer.fit(train_set, valid_set,
                    self.n_epochs, 
                     batch_size=self.batch_size,
                     print_every=self.print_every,
                     valid_every=self.valid_every,
                     num_workers=self.num_data_workers)


class SDLSTMTrainer:
    def __init__(self, normalizer_mean, normalizer_std, model, optimizer, device, prop_names, log_dir=None, clip_gradients=True) -> None:
        '''
        normalizer_mean
        normalizer_std
        model - SD LSTM model
        optimizer - optimizer, implements .step()
        device - 'cpu' or 'gpu'
        prop_names - List of property names
        '''

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.clip_gradients = clip_gradients

        # self.graph_logger = graph_logger
        self.mean = normalizer_mean
        self.std = normalizer_std

        self.prop_names = prop_names # List of property names as defined in utils/property_calculator
        self.property_cal = PropertyCalculator(self.prop_names)

        # Loss calculator
        self.loss_calc = PerpCalculator()
        # TODO: Remove old loss passing mechanism, where it's passed down as a list of losses

    def process_batch(self, batch, properties):
        '''
        Input:
         - batch: (input, target, mask)
         - properties: 
        Return:
         - loss
         - batch_size
        '''

        # TODO: Remove testing comment
        '''
        # In SMILES RNN: 
        torch.Size([20, 101])
        torch.Size([20, 1])
        torch.Size([3, 20, 512])
        torch.Size([3, 20, 512])
        2

        # Here Currently
        torch.Size([20, 99, 80])
        torch.Size([20, 1])
        torch.Size([3, 20, 512])
        torch.Size([3, 20, 512])
        2

        Difference: In the SMILES method, The prediction is the indecies of the next predicted token
        '''
        inp, tgt, mask = batch

        inp = inp.to(self.device)
        tgt = tgt.to(self.device)
        mask = mask.to(self.device)
        properties = properties.to(self.device)

        # process data
        batch_size = inp.size(0)
        hidden = self.model.init_hidden(inp.size(0), self.device)
        # Convert onehots from long to float. TODO: This could be done during data preprocessing for speed up

        # Output shape will be -> torch.Size([20, 99, 80])
        output, hidden = self.model(inp.float(), properties, hidden)

        # This line converts to form: [1980, 80]
        # For loss computation (it collapses first two dims basically)
        # output = output.view(output.size(0) * output.size(1), -1)
        '''
        print()
        print()
        print(inp.shape)
        print(properties.shape)
        print(hidden[0].shape) # Hidden layers, should be fine
        print(hidden[1].shape) # Both: [3, 20, 512]
        print(len(hidden))
        '''
        # Loss computation


        # PerpLoss expects [time_steps, batch_size, DECISION_DIM]
        # We have [batch_size, time_steps, DECISION_DIM]
        # We can convert with .permute(1, 0, 2)

        # Pre permutation: 
        #   target (int 65), loss (float32), output (float32)    [50, 99, 80] batch_size x seq_len x decision_dim
        # Post permutation:  [99, 50, 80] seq_len x batch_size x decision_dim

        loss = self.loss_calc(tgt.permute(1, 0, 2), mask.permute(1, 0, 2), output.permute(1, 0, 2))[0]
        # loss is now in form: tensor([54.0789], grad_fn=<MyPerpLossBackward>) so we take [0]
        # Original gave form: tensor(3.8469, grad_fn=<NllLossBackward0>)

        return loss, batch_size

    def train_on_batch(self, batch, properties):

        # setup model for training
        self.model.train()
        self.model.zero_grad()

        # forward / backward
        loss, size = self.process_batch(batch, properties)
        loss.backward()

        # optimize
        if self.clip_gradients:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), size

    def test_on_batch(self, batch, properties):

        # setup model for evaluation
        self.model.eval()

        # forward
        loss, size = self.process_batch(batch, properties)

        return loss.item(), size

    def batch_property_MSE(self, properties):
        '''
        properties: torch.Tensor []
        Generate a batch of molecules, compute their properties and compute the MSE of their properties compared to the target property
        '''
        self.model.eval()

        model = self.model.to(self.device)
        max_rules = model.max_rules - 1

        batch_size = properties.shape[0]

        # Prepare initial input
        initial_input = torch.zeros(batch_size, max_rules, model.rules_dict_size).to(self.device)
        initial_input[:, 0, 0] = 1

        properties = properties.to(self.device)

        hidden = model.init_hidden(batch_size, self.device)

        # Forward pass to generate raw logits
        # Since your model expects inputs at each step, you can loop through the sequence length,
        # feeding the output back as input at each step
        logits_list = []

        current_input = initial_input

        for _ in range(max_rules):
            logits, hidden = model(current_input, properties, hidden)  # Forward pass
            logits_list.append(logits[:, -1:, :])  # Append the last timestep's logits

            # Prepare the next input
            # Here, you might want to apply some sampling strategy to convert logits to discrete tokens if necessary
            # For simplicity, we're using the logits directly as the next input
            current_input = logits[:, -1:, :]

        # Concatenate the logits from each timestep to form the sequence
        raw_logits = torch.cat(logits_list, dim=1)
        
        # compute property values
        pvals = []

        # Silence RDKIT
        logger = RDLogger.logger()
        logger.setLevel(RDLogger.CRITICAL)

        smiles = raw_logit_to_smiles(np.array(raw_logits.permute(1, 0, 2)), use_random=True, quiet=True)
        
        for i in range(len(smiles)):
            try:
                mol = Chem.MolFromSmiles(smiles[i])
                pvals.append([np.array(properties[i]), self.property_cal(mol)])
                # TODO: Remove
            except:
                continue

        try:
            sum_squared_diffs = 0
            count = 0

            for true_values, predicted_values in pvals:
                squared_diffs = (true_values - predicted_values) ** 2
                sum_squared_diffs += np.sum(squared_diffs)
                count += len(true_values)
        except:
            print('Error computing property MSE')

        return sum_squared_diffs / count
   
    
    def validate_MSE(self, data_loader, n_molecule):
        """Runs validation and reports the average loss"""
        valid_losses = []
        with torch.no_grad():
            for batch_all in data_loader:
                batch = batch_all[:-1]
                properties = batch_all[-1]
                # TODO Compute prediction loss
                
                # loss, size = self.test_on_batch_MSE(batch, properties)
                loss = self.batch_property_MSE(properties)
                if loss > 0:
                    valid_losses += [loss]
                else: 
                    valid_losses = valid_losses
        return np.array(valid_losses).mean()
    
    def validate(self, data_loader, n_molecule):
        """Runs validation and reports the average loss"""
        valid_losses = []
        with torch.no_grad():
            for all_batch in data_loader:
                batch = all_batch[:-1]
                properties = all_batch[-1]
                loss, size = self.test_on_batch(batch, properties)
                valid_losses += [loss]
        return np.array(valid_losses).mean()

    def train_extra_log(self, n_molecules):
        pass

    def valid_extra_log(self, n_molecules):
        pass

    def fit(self, training_data, test_data, n_epochs, batch_size, print_every,
            valid_every, num_workers=0):
        training_round = _ModelTrainingRound(self, training_data, test_data, n_epochs, batch_size, print_every,
                                             valid_every, num_workers)
        return training_round.run()



class _ModelTrainingRound:
    """
    Performs one round of model training.

    Is a separate class from ModelTrainer to allow for more modular functions without too many parameters.
    This class is not to be used outside of ModelTrainer.
    """
    class EarlyStopNecessary(Exception):
        pass

    def __init__(self, model_trainer: SDLSTMTrainer, training_data, test_data, n_epochs, batch_size, print_every,
                 valid_every, num_workers=0) -> None:
        self.model_trainer = model_trainer
        self.training_data = training_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.valid_every = valid_every
        self.num_workers = num_workers

        self.start_time = time()
        self.unprocessed_train_losses: List[float] = []
        self.all_train_losses: List[float] = []
        self.all_valid_losses: List[float] = []
        self.n_molecules_so_far = 0
        self.has_run = False
        self.min_valid_loss = np.inf
        self.min_avg_train_loss = np.inf
        
        self.iter = 0

    def run(self):
        if self.has_run:
            raise Exception('_ModelTrainingRound.train() can be called only once.')

        try:
            for epoch_index in range(1, self.n_epochs + 1):
                self._train_one_epoch(epoch_index)

            self._validation_on_final_model()
        except _ModelTrainingRound.EarlyStopNecessary:
            print('ERROR: Probable explosion during training. Stopping now.')

        self.has_run = True
        return self.all_train_losses, self.all_valid_losses

    def _train_one_epoch(self, epoch_index: int):
            # with profiler.profile() as prof:

            print(f'EPOCH {epoch_index}')
            # shuffle at every epoch
            data_loader = DataLoader(self.training_data,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=self.num_workers,
                                    pin_memory=True)

            epoch_t0 = time()
            self.unprocessed_train_losses.clear()

            for batch_index, batch_all in enumerate(data_loader):
                batch = batch_all[:-1]
                properties = batch_all[-1]
                self._train_one_batch(batch_index, batch, properties, epoch_index, epoch_t0)
            
            # report validation progress?

            if epoch_index % self.valid_every == 0:
                self._report_validation_progress(epoch_index)
        
            # profiler.emit_nvtx()
            # print('TODO: Remove profiler!!!!')
            # print(prof.key_averages().table(sort_by="self_cpu_time_total"))


    def _train_one_batch(self, batch_index, batch, properties, epoch_index, train_t0):
        loss, size = self.model_trainer.train_on_batch(batch, properties)

        self.unprocessed_train_losses += [loss]
        self.n_molecules_so_far += size

        # report training progress?
        if batch_index > 0 and batch_index % self.print_every == 0:
            self._report_training_progress(batch_index, epoch_index, epoch_start=train_t0)



    def _report_training_progress(self, batch_index, epoch_index, epoch_start):
        mols_sec = self._calculate_mols_per_second(batch_index, epoch_start)

        # Update train losses by processing all losses since last time this function was executed
        avg_train_loss = np.array(self.unprocessed_train_losses).mean()
        self.all_train_losses += [avg_train_loss]
        self.unprocessed_train_losses.clear()

        # Logger
        print('TRAIN | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch|batch : {epoch_index}|{batch_index} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'mols/sec: {mols_sec:.2f} | '
            f'train_loss: {avg_train_loss:.4f}')

        self.iter = self.iter + 1
        self.model_trainer.train_extra_log(self.n_molecules_so_far)

        self._check_early_stopping_train_loss(avg_train_loss)
        # self.model_trainer.graph_logger.add_scalar('average_trainingloss', avg_train_loss, global_step = self.iter, save_csv= True)
        print(f'Average Training Loss: {avg_train_loss}')

    def _calculate_mols_per_second(self, batch_index, epoch_start):
        """
        Calculates the speed so far in the current epoch.
        """
        train_time_in_current_epoch = time() - epoch_start
        processed_batches = batch_index + 1
        molecules_in_current_epoch = self.batch_size * processed_batches
        return molecules_in_current_epoch / train_time_in_current_epoch

    def _report_validation_progress(self, epoch_index):
        avg_valid_loss = self._validate_current_model()
        # self.model_trainer.graph_logger.add_scalar('average_validloss', avg_valid_loss, global_step = self.iter, save_csv= True)

        print(f'VALID | Prop-Loss | {avg_valid_loss}')

        self._log_validation_step(epoch_index, avg_valid_loss)
        self._check_early_stopping_validation(avg_valid_loss)

        # If best so far in terms of prop reconstruction or we're at an even epoch number
        if self.model_trainer.log_dir:
            if (avg_valid_loss <= min(self.all_valid_losses) and avg_valid_loss > 0) or (epoch_index % 5 == 0):
                self._save_current_model(self.model_trainer.log_dir, epoch_index, avg_valid_loss)

    def _validate_current_model(self):
        """
        Validate the current model.

        Returns: Validation loss.
        """
        test_loader = DataLoader(self.test_data,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 pin_memory=True)
        avg_valid_loss = self.model_trainer.validate_MSE(test_loader, self.n_molecules_so_far)
        self.all_valid_losses += [avg_valid_loss]
        return avg_valid_loss

    def _log_validation_step(self, epoch_index, avg_valid_loss):
        """
        Log the information about the validation step.
        """
        print(
            'VALID | '
            f'elapsed: {time_since(self.start_time)} | '
            f'epoch: {epoch_index}/{self.n_epochs} ({self._get_overall_progress():.1f}%) | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {avg_valid_loss:.4f}')

        self.model_trainer.valid_extra_log(self.n_molecules_so_far)
        # logger.info('')

    def _get_overall_progress(self):
        total_mols = self.n_epochs * len(self.training_data)
        return 100. * self.n_molecules_so_far / total_mols

    def _validation_on_final_model(self):
        """
        Run validation for the final model and save it.
        """
        valid_loss = self._validate_current_model()
        print(
            'VALID | FINAL_MODEL | '
            f'elapsed: {time_since(self.start_time)} | '
            f'molecules: {self.n_molecules_so_far} | '
            f'valid_loss: {valid_loss:.4f}')

        if self.model_trainer.log_dir:
            self._save_model(self.model_trainer.log_dir, 'final', valid_loss)

    def _save_current_model(self, base_dir, epoch, valid_loss):
        """
        Delete previous versions of the model and save the current one.
        """
        for f in glob(os.path.join(base_dir, 'model_*')):
            os.remove(f)

        self._save_model(base_dir, epoch, valid_loss)

    def _save_model(self, base_dir, info, valid_loss):
        """
        Save a copy of the model with format:
                model_{info}_{valid_loss}
        """
        base_name = f'LSTM_{info}_{valid_loss:.3f}'
        print(base_name)
        save_model(self.model_trainer.model, base_dir, base_name)

    def _check_early_stopping_train_loss(self, avg_train_loss):
        """
        This function checks whether the training has exploded by verifying if the avg training loss
        is more than 10 times the minimal loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 10 * self.min_avg_train_loss
        if avg_train_loss > threshold:
            raise _ModelTrainingRound.EarlyStopNecessary()

        # update the min train loss if necessary
        if avg_train_loss < self.min_avg_train_loss:
            self.min_avg_train_loss = avg_train_loss

    def _check_early_stopping_validation(self, avg_valid_loss):
        """
        This function checks whether the training has exploded by verifying if the validation loss
        has more than doubled compared to the minimum validation loss so far.

        If this is the case, a EarlyStopNecessary exception is raised.
        """
        threshold = 2 * self.min_valid_loss
        if avg_valid_loss > threshold:
            raise _ModelTrainingRound.EarlyStopNecessary()

        if avg_valid_loss < self.min_valid_loss:
            self.min_valid_loss = avg_valid_loss


