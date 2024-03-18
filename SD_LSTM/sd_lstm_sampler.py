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

from torch.distributions import Categorical

from tree_walker import ConditionalDecoder
from attribute_tree_generator import create_sequential_tree_decoder, FETCH_TOKEN, RETURN_TOKEN

from mol_tree import Node, get_smiles_from_tree


FAILED_TO_TERM_ERR_STR = 'ERROR-STILL-DECODING'
FAILED_TO_DECODE_ERR_STR = 'ERROR-FAILED-TO-CONSTRUCT-SMILE'

class SDLSTMSampler:
    def __init__(self, batch_size, device, rules_dict_size, distribution_cls = Categorical):
        self.batch_size = batch_size
        self.device = device
        self.rules_dict_size = rules_dict_size
        self.distribution_cls = distribution_cls


    def pad_one_hots(self, one_hots, max_rules):
        padding_length = max_rules - one_hots.shape[0]
        if padding_length > 0:
                # Create a one-hot encoded vector for the padding token
            padding_vector = np.zeros((1, one_hots.shape[1]), dtype=np.float32)  # Initialize with zeros
            padding_vector[0, model.rules_dict_size - 1] = 1  # Set the appropriate index for the padding token to 1

            # Create a padding array with the desired padding length
            padding_array = np.tile(padding_vector, (padding_length, 1))  # Repeat the padding vector

            # Append the padding array to the original sequence of one-hots
            padded_one_hots = np.concatenate((one_hots, padding_array), axis=0)
            return padded_one_hots
        else:
            return one_hots


    def sample_batch_masking_smiles(self, model, max_rules, property_val):
        '''
        Sample a batch of smiles with property values described in property_val. If max_decoding limit reached without
        termination return ERROR instead of that smiles
        
        - model : model
        - max_rules : int max decoding steps
        - property_val : torch.Tensor of shape num_to_sample x num_props containing properties 
        return smiles (arr str containing sampled smiles)
        '''
        assert len(property_val.shape) == 2 and property_val.shape[1] == model.property_size


        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            # Infer batch size
            b_size = property_val.shape[0]

            # Initialize model inputs and send to device
            model = model.to(self.device)
            property_values = torch.Tensor(property_val).to(self.device)
            hidden = model.init_hidden(b_size , self.device)
            # Current input for model
            
            # Initialize a tensor of zeros with the shape [b_size, model.rules_dict_size]
            current_input = torch.zeros([b_size, 1 , model.rules_dict_size], dtype=torch.float, device=self.device)

            # Set the first element of each row in current_input to 1
            current_input[:, 0] = 1
            
            # Array fo walkers, one for each element of batch_size
            still_executing = [True] * b_size
            walkers = []
            root_nodes = []
            dec_generators = []
            for _ in range(b_size):
                # INIT WALKERS
                onehot_start = torch.zeros(1, model.rules_dict_size)
                onehot_start[0][0] = 1
                new_walker = ConditionalDecoder(onehot_start, use_random=False)
                walkers.append(new_walker)
                new_node = Node('smiles')
                root_nodes.append(new_node)
                tree_decoder = create_sequential_tree_decoder()
                dec_gen = tree_decoder.decode(new_node, new_walker)
                # Burn initial logit / input
                fmsk, frsp = next(dec_gen)
                dec_generators.append(dec_gen)
                # TODO: Remove this once I'm confident it's working properly
                assert frsp == FETCH_TOKEN and fmsk == [0]

            # TODO: Remove Debugging code
            tokens = [[] for _ in range(b_size)]

            for rindex in range(max_rules):
                '''
                PROCESS:
                1. Next to get next mask
                2. Compute logits, mask them and add them to the walker
                3. Continue execution
                '''
                logits, hidden = model(current_input, property_values, hidden)
                # Fetch Next Mask
                # Fetch next masks
                for ind in range(b_size):
                    if not still_executing[ind]:
                        # Already finished executing
                        continue
                    # Get mask and check still running
                    
                    # TODO: Remove debugging code
                    # print(tokens[ind])

                    mind, resp = next(dec_generators[ind])
                    if resp == RETURN_TOKEN:
                        still_executing[ind] = False
                        # We've just finished, no need to sample any more
                        continue

                    mask = torch.zeros(model.rules_dict_size, dtype=torch.bool)
                    mask[mind] = True
                    cur_step_logits = logits[ind][0]
                    # Logits at step nexst index
                    cur_step_logits[~mask] = float('-inf')

                    # Compute probabilities
                    cur_exped = np.exp(cur_step_logits - cur_step_logits.max())
                    cur_prob = cur_exped / (cur_exped.sum() + 1e-8)

                    distribution = self.distribution_cls(probs=cur_prob)
                    new_tok = distribution.sample().item()
                    new_log = torch.zeros(1, model.rules_dict_size)
                    new_log[0][new_tok] = 1

                    walkers[ind].raw_logits = np.append(walkers[ind].raw_logits, new_log, axis=0)

                    current_input[ind] = new_log.clone().detach()
                    
                    # TODO: REMOVE DEBUGGING CODE
                    # tokens[ind].append(new_tok)


        smiles = []
        # Compute smiles
        for ind in range(b_size):
            # If hasn't finished decoding, set to error
            if still_executing[ind]:
                smiles.append(FAILED_TO_TERM_ERR_STR)
            else:
                try:
                    smiles.append(get_smiles_from_tree(root_nodes[ind]))
                except:
                    smiles.append(FAILED_TO_DECODE_ERR_STR)
        return smiles

    def sample(self, model, properties, num_to_sample, max_seq_len, quiet=True):
        '''
        Interface for evaluation code
        '''
        assert properties.shape[0] == num_to_sample

        properties = model.normalize_prop_scores(properties)

        smiles = []  # finished smiles
        num_batches = (num_to_sample + self.batch_size - 1) // self.batch_size

        if quiet:
            itr = range(num_batches)
        else:
            itr = tqdm(range(num_batches))

        for batch_idx in itr:
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_to_sample)

            properties_batch = properties[start_idx:end_idx]
            batch_smiles = self.sample_batch_masking_smiles(model, max_seq_len, properties_batch)
            smiles.extend(batch_smiles)

        return smiles

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
    # model_weights = '../models/SD_LSTM_FR/SD_LSTM_silent-queen-29_Epoch_5_Vl_0.123.pt'
    # model_definit = '../models/SD_LSTM_FR/SD_LSTM_silent-queen-29_Epoch_5_Vl_0.123.json'
    model_weights = '../models/SD_LSTM_QM9_MASKED_CROSS_ENTROPY_TOKENS/SD_LSTM_dark-hall-89_Epoch_20_Vl_0.201.pt'
    model_definit = '../models/SD_LSTM_QM9_MASKED_CROSS_ENTROPY_TOKENS/SD_LSTM_dark-hall-89_Epoch_20_Vl_0.201.json'

    sampler = SDLSTMSampler(batch_size = 128, device = 'cpu', rules_dict_size = 80)
    model = load_model(model_definit, model_weights, device='cpu')

    max_rules = 100
    property_val = -3

    n_props = 100
    import random
    # random.uniform(-3, 3)
    properties = torch.tensor([[property_val] for _ in range(n_props)])
    properties = properties.to(torch.float32)

    smiles = sampler.sample(model, properties, n_props, 100 + 50, quiet=False)
    # Print non terminated
    non_termed = smiles.count(FAILED_TO_TERM_ERR_STR)
    print(f'Didnt terminate: {non_termed}')
    # Print other error
    failed_decode = smiles.count(FAILED_TO_DECODE_ERR_STR)
    print(f'Failed to decode: {failed_decode}')

    unique_pct = len(set(smiles)) / len(smiles)
    print(f'Unique PCT {unique_pct}')

    calprops = props_from_smiles(smiles)
    avg_prop = np.mean(calprops)
    print(f'{avg_prop} target: {property_val}')