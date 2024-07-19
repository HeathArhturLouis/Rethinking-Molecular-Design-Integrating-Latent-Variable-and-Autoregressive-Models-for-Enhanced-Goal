'''
Action sampler is slow, I think I can write a faster sampler in the time it takes for it to run
~Louis
'''

from rnn_model import ConditionalSmilesRnn
import math

import torch.nn.functional as F
import torch

import sys
import numpy as np

from rnn_utils import load_rnn_model, rnn_start_token_vector

import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)) , '../utils'))

from smiles_char_dict import SmilesCharDictionary
from torch.distributions import Categorical

class FastSampler:
    def __init__(self, device = 'cpu', batch_size=64):
        self.device = 'cpu'
        self.batch_size = batch_size
        self.sd = SmilesCharDictionary()

        # Class of distribution for sampling logits from
        self.distribution_cls = Categorical


    def sample_batch_actions(self, model: ConditionalSmilesRnn, properties, max_seq_len=100, random_sample = False):
        '''
        properties -- torch tensor containing self.batch_size property values
        '''
        assert properties.shape[0] == self.batch_size
        max_seq_len += 1 # As in training set for end token

        with torch.no_grad():
            # We will set the initial token
            max_rules = max_seq_len - 1

            # Prepare initial inputs: using zeros as initial inputs
            # Batch_size x 1
            # initial_input = torch.Tensor([ [self.sd.char_idx[self.sd.BEGIN]] for i in range(self.batch_size)]).to(self.device)

            # torch tensor of [1]'s of len batch_len
            initial_input = rnn_start_token_vector(self.batch_size, self.device)

            hidden = model.init_hidden(self.batch_size, self.device)

            # Store sampled tokens for batch
            # Actions is shape batch_size x max_seq_len and has each first row element set to 1
            actions = torch.zeros((self.batch_size, max_seq_len), dtype=torch.long).to(self.device)

            # TODO: Is this neccesary?
            current_input = initial_input

            # Turns out the decoder was handling this all along .... LOUIS TODO: Remove dead code
            # Handle terminated strings so it doesn't add characters post termination
            # termed = torch.ones(self.batch_size, dtype=torch.long) # will equal 1 for unterminated sequences
            
            actions, hidden = model.forward(x=None, properties=properties, hidden=hidden, use_teacher_forcing = False, sampling = random_sample, return_actions = True)

            '''
            # TODO: Is this neccesary
            for i in range(0, max_seq_len):
                # Input:
                logits, hidden = model(x=current_input, properties=properties, hidden=hidden)  # Forward pass
                # Logits shape is batch_size x 1 x token_dict_size

                # convert to probabilities and sample next action
                prob = F.softmax(logits, dim=2)

                distribution = self.distribution_cls(probs=prob)
                action = distribution.sample()
                action = action.squeeze()

                # Mask actions for already terminated smiles
                # action *= termed
                # Update termination mask to include smile sthat terminated this iteration

                # termed[action == self.sd.char_idx[self.sd.END]] = 0
                # Add action to end of string, sequeeze makes actions size batch_size
                actions[:, i] = action
                
                # Actions is ints for next selected actions 
                # Actions is batch_size x 1

                # Input for next iteration is next action
                current_input = action.unsqueeze(dim=1)
            '''
            return actions

    def sample_batch_smiles(self, model: ConditionalSmilesRnn, properties, max_seq_len=100, random_sample=False):
        indecies = self.sample_batch_actions(model, properties, max_seq_len, random_sample=random_sample)

        indecies = np.array(indecies)
        # print(indecies.reshape(-1, max_seq_len + 2).shape)
        return np.array(self.sd.matrix_to_smiles(indecies))



    def sample(self, model: ConditionalSmilesRnn, properties, num_to_sample : int, max_seq_len = 100, random_sample = False):
        '''
        sample num to smaple smiles with properties property values
        '''
        assert len(properties) == num_to_sample

        model.eval()

        no_batches = num_to_sample // self.batch_size

        smiles = []

        # I need to handle termination, i.e. stop generating a sequence past when the end character is generated
        for i in range(no_batches):
            b_props = properties[i * self.batch_size : (i + 1) * self.batch_size, :]
            nts = len(b_props)

            smiles += list(self.sample_batch_smiles(model=model, properties=b_props , max_seq_len = max_seq_len, random_sample = random_sample))

        remainder = num_to_sample % self.batch_size


        if remainder != 0:
            dummy_props = torch.zeros([self.batch_size, 1])
            dummy_props[:remainder] = properties[(no_batches) * self.batch_size :]

            rem_smiles = self.sample_batch_smiles(model, dummy_props, max_seq_len)
            smiles += list(rem_smiles[:remainder])
            if len(rem_smiles) != self.batch_size:
                print('Warning to self: One or more geneations failed!' )
        else:
            rem_smiles = []
        #Warning to myself so I can know this is happening
        if len(smiles) != ((no_batches) * self.batch_size) + remainder:
            print('Warning to self: One or more geneations failed!')
        return smiles[:num_to_sample]


if __name__ == '__main__':
    model_weights = '../models/LSTM_QM9/batch_size_20_2/LSTM_8_1.151.pt'
    model_definit = '../models/LSTM_QM9/batch_size_20_2/LSTM_8_1.151.json'

    model_weights = '../models/LSTM_TF_03/LSTM_12_1.180.pt'
    model_definit = '../models/LSTM_TF_03/LSTM_12_1.180.json'


    # TODO: Check if batch size mattters at all, i assume not
    sampler = FastSampler(batch_size = 64, device = 'cpu')

    model = load_rnn_model(model_definit, model_weights, 'cpu', copy_to_cpu=True)

    nts = 1000
    prange = [-3, 3]

    # properties = torch.Tensor([np.random.uniform(prange[0], prange[1], size=1) for i in range(nts)])

    # act = sampler.sample_batch_actions(model=model, num_to_sample = 64, properties = properties, max_seq_len = 100)

    # print(sampler.sample_batch_smiles(model=model, num_to_sample = 64, properties = properties, max_seq_len = 100))
    # print(sampler.sample(model=model, num_to_sample = nts, properties=properties, max_seq_len = 100))
    targets = np.random.uniform(low=-3, high=3, size=nts)
    sampler.sample(model=model, properties=torch.Tensor([[a] for a in targets]), num_to_sample=nts, max_seq_len=100)
