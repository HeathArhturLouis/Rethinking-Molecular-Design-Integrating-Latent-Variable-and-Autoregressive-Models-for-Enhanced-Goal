import logging
from typing import List
import random
import torch
import numpy as np
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

from rnn_model import ConditionalSmilesRnn
from rnn_trainer import SmilesRnnTrainer
from rnn_utils import get_tensor_dataset, load_smiles_and_properties, set_random_seed

from smiles_char_dict import SmilesCharDictionary

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

# TODO: hard-coded to 9 properties?
PROPERTY_SIZE = 1

class SmilesRnnDistributionLearner:
    def __init__(self, data_set, output_dir, n_epochs, hidden_size=512, n_layers=3,
                 max_len=100, batch_size=64, rnn_dropout=0.2, lr=1e-3, valid_every=100, prop_model=None) -> None:
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
        self.print_every = 10
        self.prop_model = prop_model
        self.seed = 42

    def train(self, data_path):
        # GPU if available
        cuda_available = torch.cuda.is_available()
        device_str = 'cuda' if cuda_available else 'cpu'
        device = torch.device(device_str)
        print(f'CUDA enabled:\t{cuda_available}')

        set_random_seed(self.seed, device)
        
        property_names, train_x, train_y, valid_x, valid_y = load_smiles_and_properties(data_path)
        n_props = len(property_names)

        #Louis: load_smiles_and_properties now takes a datapath to data dir and returns all needed dataset components
        #train_seqs, train_prop = alid_y = load_smiles_and_properties(training_set, False)
        #sample_indexs = np.arange(train_seqs.shape[0])
        #random.shuffle(sample_indexs)

        #Compatible validation sets
        #train_x, train_y = train_seqs[10000:,:], train_prop[10000:,:]
        #valid_x, valid_y = train_seqs[:10000,:], train_prop[:10000,:]
        '''
        if self.prop_model is not None:
            train_y = self.prop_model.transform(train_y)
            valid_y = self.prop_model.transform(valid_y)
        '''
        
        #LOUIS: TODO: Is this neccesary? Since some of the properties have already been normalized?
        #scale the property to fall between -1 and 1
        all_y = np.concatenate((train_y, valid_y), axis=0)
        mean = np.mean(all_y, axis = 0)
        std = np.std(all_y, axis = 0)
        #np.save(data_path + '/normalizer.py', [mean, std])
        train_y = (train_y - mean) / std
        valid_y = (valid_y - mean) / std
        
        # convert to torch tensor, input, output smiles and properties    
        train_set = get_tensor_dataset(train_x, train_y)  # lstm_params = '../my_code/models/NEW_LONG_RUNS/ZINC120/LSTM/LSTM_9_0.486.pt'
        valid_set = get_tensor_dataset(valid_x, valid_y)



        sd = SmilesCharDictionary()
        n_characters = sd.get_char_num()

        # build network
        smiles_model = ConditionalSmilesRnn(input_size=n_characters,
                                            property_size=n_props, #PROPERTY_SIZE,
                                            # property_names=property_names, # Record names of properties for future use 
                                            hidden_size=self.hidden_size,
                                            output_size=n_characters,
                                            n_layers=self.n_layers,
                                            rnn_dropout=self.rnn_dropout)

        # wire network for training
        optimizer = torch.optim.Adam(smiles_model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=sd.pad_idx)

        trainer = SmilesRnnTrainer(normalizer_mean = mean,
                                   normalizer_std = std,
                                   model=smiles_model,
                                   criteria=[criterion],
                                   optimizer=optimizer,
                                   device=device,
                                   # prop_names = property_names,
                                   log_dir=self.output_dir) 

        trainer.fit(train_set, valid_set,
                    self.n_epochs, 
                     batch_size=self.batch_size,
                     print_every=self.print_every,
                     valid_every=self.valid_every,
                     num_workers=0)
