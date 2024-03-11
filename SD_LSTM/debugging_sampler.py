import torch
import numpy as np
import math

# from typing import Type

import torch.nn.functional as F

from sd_lstm_model import ConditionalSDLSTM
from sd_lstm_utils import load_model

from torch import nn

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)) , '../utils'))
from sd_smiles_decoder import raw_logit_to_smiles


class SDLSTMSampler:
    def __init__(self, batch_size, device, rules_dict_size):
        self.batch_size = batch_size
        self.device = device
        self.rules_dict_size = rules_dict_size


    def sample_batch_actions(self, model, max_rules, property_values):
        '''
        Autoregressively sample a batch of logits from the model without enforcing rule validity at each step.

        device: string device ('cpu' | 'gpu')
        Properties: torch.Tensor([List of pvals for all generations) or torch.Tensor(single prop value to be repeated)
        batch_size: size of batch to sample
        '''

        # TODO: Move to outer method
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
          
        model = model.to(self.device)
        property_values = property_values.to(self.device)

        with torch.no_grad():
            model = model.to(self.device)
            max_rules -= 1
            # Sequence length dimension is changed to 1, since we're feeding in timesteps 1 at a time
            # If this was for example 2, my guess is the model would likely step twice
            initial_input = torch.zeros(self.batch_size, 1, self.rules_dict_size).to(self.device)
            initial_input[:, 0 , 0] = 1

            assert property_values.shape[0] == self.batch_size 

            hidden = model.init_hidden(self.batch_size, self.device)
            actions_list = []
            current_input = initial_input

            for i in range(max_rules):
                logits, hidden = model(current_input, property_values, hidden)  # Forward pass
                # Logits is shape [batch_size x 1 x decision_dim] 
                # and logits 
                probs = torch.softmax(logits[:, -1, :], dim=1)
                samples = torch.multinomial(probs, 1)
                # samples is batch_size x 1 <-- indecies of samples

                # Create zeros tenso of shape as logits, scatter sets the indecies of sample indecies to 1
                current_input = torch.zeros_like(logits).scatter_(-1, samples.unsqueeze(-1), 1)
                
                # actions is shape [batch_size, 1, decision_dim]
                actions_list.append(current_input)  # Append the last timestep's logits


            # Concatenate the logits from each timestep to form the sequence
            actions_list = torch.cat(actions_list, dim=1)

            return actions_list


    def sample_batch_unconstrained_logits(self, model, max_rules, property_values):
        '''
        Autoregressively sample a batch of logits from the model without enforcing rule validity at each step.

        device: string device ('cpu' | 'gpu')
        Properties: torch.Tensor([List of pvals for all generations) or torch.Tensor(single prop value to be repeated)
        batch_size: size of batch to sample
        '''

        # model.eval()
        with torch.no_grad():
            # TODO: Repeat for number of samples
            model = model.to(self.device)

            # We will set the initial token
            max_rules -= 1

            # Prepare initial inputs: using zeros as initial inputs
            initial_input = torch.zeros(self.batch_size, max_rules, self.rules_dict_size).to(self.device)

            # Set first index to initial token
            initial_input[:, 0, 0] = 1
            
            assert property_values.shape[0] == self.batch_size 

            hidden = model.init_hidden(self.batch_size, self.device)

            # Forward pass to generate raw logits
            # Since your model expects inputs at each step, you can loop through the sequence length,
            # feeding the output back as input at each step
            logits_list = []

            current_input = initial_input

            for _ in range(max_rules):
                logits, hidden = model(current_input, property_values, hidden)  # Forward pass
                logits_list.append(logits[:, -1:, :])  # Append the last timestep's logits

                # Prepare the next input
                # Here, you might want to apply some sampling strategy to convert logits to discrete tokens if necessary
                # For simplicity, we're using the logits directly as the next input
                current_input = logits[:, -1:, :]

            # Concatenate the logits from each timestep to form the sequence
            raw_logits = torch.cat(logits_list, dim=1)

            return raw_logits.detach()


    def sample_batch_unconstrained_smiles(self, model, max_rules, property_values, quiet=False):
        '''
        Generate logits autoregressively, then enforce constraints during parse time

        Return:
        One batch of smiles strings
        '''
        logits = self.sample_batch_unconstrained_logits(model = model, max_rules = max_rules, property_values = property_values)
        # input expected to be array
        return raw_logit_to_smiles(np.array(logits.permute(1, 0, 2)), use_random=True, quiet=quiet)


    def sample(self, model, properties, num_to_sample, max_seq_len, quiet=True, force = False):
        '''
        mode : 'UNC' for unconstrained logit generation OR 'CONST' for constrained sequential generation
        properties : torch.Tensor[[prop_value to be repeated]] or torch.Tensor([[properties] * num_to_sample])
        Wrapper function for compatibility with vanilla ConditionalLSTM sampler
        '''

        if not force:
            n_batches = math.ceil(num_to_sample / self.batch_size)
            smiles = []
            # for batch in range(n_batches):
            for i in range(n_batches):
                smiles += self.sample_batch_unconstrained_smiles(model=model, max_rules = max_seq_len, property_values=properties[0], quiet=quiet)
        else:
            smiles = []
            while len(smiles) < num_to_sample:
                smiles += self.sample_batch_unconstrained_smiles(model=model, max_rules = max_seq_len, property_values=properties[0], quiet=quiet)

        return smiles[:num_to_sample]


from rdkit import Chem
from tqdm import tqdm

import sys
sys.path.append('../utils/')
from property_calculator import PropertyCalculator
pc = PropertyCalculator(['LogP'])

def props_from_smiles(smiles_list, verbose=True):
    '''
    Computes property scores of all valid smiles and returns as list
    '''
    if verbose:
        smiles_list = tqdm(smiles_list)
    props = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            props.append(pc(mol)[0])
    return props


if __name__ == '__main__':
    # Lets assume this model is heavity overfit
    # model_weights = '../models/SD_LSTM_QM9/batch_size_20_1/LSTM_20_1.682.pt'
    # model_definit = '../models/SD_LSTM_QM9/batch_size_20_1/LSTM_20_1.682.json'

    '''
    
    model_weights = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_6_1.559.pt'
    model_definit = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_6_1.559.json'
    
    sampler = SDLSTMSampler(batch_size = 128, device = 'cpu', rules_dict_size = 80)
    model = load_model(model_definit, model_weights, device='cpu')

    
    # Try binary CE loss

    actions = sampler.sample_batch_actions(model=model, max_rules=100, property_values=torch.Tensor([[3] for _ in range(128) ]))
    logits = sampler.sample_batch_unconstrained_logits(model=model, max_rules=100, property_values=torch.Tensor([[3] for _ in range(128)]))

    # Acrtions is batch_size x num_steps x decision dim
    print(torch.argmax(actions[50], dim=1))
    print(torch.argmax(actions[27], dim=1))
    print(logits[0][0])
    probs = torch.softmax(logits[0][0], dim=0)
    print(probs)
    print('Looking for degenerate probabilities:')

    print(probs[55])
    print(logits[0][0][55])

    print(probs[5])
    print(logits[0][0][5])

    print(actions.shape)
    print(np.sum(actions == 1))
    sys.exit()
    '''

    

    # I need to learn to make trees not from SMILES but from rules
    grammar = cfg_parser.Grammar(CONFIG.grammar_file)

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = cfg_parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
    onehot, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

    return (onehot, masks)


    
    model_weights = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_30_1.473.pt'
    model_definit = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_30_1.473.json'


    sampler = SDLSTMSampler(batch_size = 128, device = 'cpu', rules_dict_size = 80)
    model = load_model(model_definit, model_weights, device='cpu')

    logits = sampler.sample_batch_unconstrained_logits(model=model, max_rules=100, property_values= torch.Tensor([[1.5]for _ in range(128)]))

    print(logits[0])
    print()
    print(logits[1])

    '''
    num_to_sample = 128
    property_val = 2.5 # -2.1 # 1.5 # -2.1# Pick the most likely to fail
    properties = torch.Tensor([[property_val] for i in range(num_to_sample)])

    invalid = []
    failed_to_gen = []
    avg_prop_score = []

    for i in tqdm(range(10)):
        smiles = sampler.sample_batch_unconstrained_smiles(model=model, max_rules=100, property_values=properties)
        #smiles = sampler.sample(model=model, properties=properties, num_to_sample=num_to_sample, max_seq_len=100)
        comp_props = props_from_smiles(smiles)
        failed_to_gen.append(num_to_sample - len(smiles))
        invalid.append(len(smiles) - len(comp_props))

        avg_prop_score.append(np.nanmean(comp_props))
        print(smiles)

    print(f'Number Failed to generate : {np.mean(failed_to_gen)}')
    print(f'Number invalid : {np.nanmean(invalid)}')
    print(f'Avg prop score: {np.nanmean(avg_prop_score)} | (target: {property_val})')
    '''

'''
Mean perf for target = -2.1

VANILLA:
Number Failed to generate : 82.6
Number invalid : 9.6
Avg prop score: 0.8197438424746817 | (target: -2.1)


BINARY CROSS ENTROPY:
'''