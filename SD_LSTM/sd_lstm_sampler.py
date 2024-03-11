import torch
import numpy as np
import math

# from typing import Type

import torch.nn.functional as F

from sd_lstm_model import ConditionalSDLSTM
from sd_lstm_utils import load_model

import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)) , '../utils'))
from sd_smiles_decoder import raw_logit_to_smiles


class SDLSTMSampler:
    def __init__(self, batch_size, device, rules_dict_size):
        self.batch_size = batch_size
        self.device = device
        self.rules_dict_size = rules_dict_size

    '''
    def sample_batch_constrained(self, model, max_rules, property_values):
        # Autoregressively sample a batch of smiles enforcing rule validity at each step 
        # model: SDLSTM model
        # max_rules: maximum number of grammar rules (sequence length before SMILES decoding)
        # property_values: torch.Tensor( [list of property values])
        
        model.eval()
        torch.no_grad()

        model = model.to(self.device)
        max_rules -= 1

        initial_input = torch.zeros(self.batch_size, 1, self.rules_dict_size).to(self.device)
        initial_input[:, 0, 0] = 1

        properties = property_values.repeat(self.batch_size, 1).to(self.device)
        hidden = model.init_hidden(self.batch_size, self.device)

        rules_list = []
        logits_list = []
        current_input = initial_input

        for _ in range(max_rules):
            # Get logits for next step
            
            # Logits shape is torch.Size([64, _, 80]), next char logits are final index
            logits, hidden = model(current_input, properties, hidden)

            # Final index is logits for next step
            next_step_logits = logits[:, -1, :]

            walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)

            # Check validity


    for i in range(raw_logits.shape[1]):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)
        new_t = Node('smiles')
        try:
            tree_decoder = create_tree_decoder()
            tree_decoder.decode(new_t, walker)
            sampled = get_smiles_from_tree(new_t)
            
        except Exception as ex:
            if not type(ex).__name__ == 'DecodingLimitExceeded':
                print('Warning, decoder failed with', ex)

            # failed. output None
            sampled = None
 
        if sampled is None:
            continue
        
        # we generated a good one
        generations.append(sampled)


            # logits_list.append(logits[:, -1:, :])

            print()
            print()
            print(logits.shape)
            print(logits)
            print()
            print()
            print(logits_list)

            return
            # Check validity and 
            
            # Construct next input
            current_input = logits[:, -1:, :]

        # Concatenate the logits from each timestep to form the sequence
        raw_logits = torch.cat(logits_list, dim=1)
    '''


    def sample_batch_unconstrained_logits(self, model, max_rules, property_values):
        '''
        Autoregressively sample a batch of logits from the model without enforcing rule validity at each step.

        device: string device ('cpu' | 'gpu')
        Properties: torch.Tensor([List of pvals for all generations) or torch.Tensor(single prop value to be repeated)
        batch_size: size of batch to sample
        '''

        model.eval()
        with torch.no_grad():
            # TODO: Repeat for number of samples
            model = model.to(self.device)

            # We will set the initial token
            max_rules -= 1

            # Prepare initial inputs: using zeros as initial inputs
            initial_input = torch.zeros(self.batch_size, max_rules, self.rules_dict_size).to(self.device)

            # Set first index to initial token
            initial_input[:, 0, 0] = 1
            
            # Create properties array
            if  property_values.shape[0] == 1:
                properties = property_values.repeat(self.batch_size, 1).to(self.device)
            else:
                assert property_values.shape[0] == self.batch_size 
            

            hidden = model.init_hidden(self.batch_size, self.device)

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


    def sample(self, model, properties, num_to_sample, max_seq_len, mode='UNC', quiet=True, force = False):
        '''
        mode : 'UNC' for unconstrained logit generation OR 'CONST' for constrained sequential generation
        properties : torch.Tensor[[prop_value to be repeated]] or torch.Tensor([[properties] * num_to_sample])
        Wrapper function for compatibility with vanilla ConditionalLSTM sampler
        '''
        assert mode in ['UNC', 'CONST']

        if mode == 'CONST':
            raise Exception('Constrained sampling not implemented. you should implement it TODO:')

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



if __name__ == '__main__':
    model_weights = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_40_1.489.pt'
    model_definit = '../models/SD_LSTM_QM9_BCE/batch_size_64_1/LSTM_40_1.489.json'

    sampler = SDLSTMSampler(batch_size = 64, device = 'cpu', rules_dict_size = 80)

    model = load_model(model_definit, model_weights, device='cpu')

    print(sampler.sample( model=model, num_to_sample=100,  properties= torch.Tensor([[2.5] for _ in range(100)]),max_seq_len=100 ))

    #print(sampler.sample_batch_unconstrained_logits(model=model, max_rules=100, property_values=torch.Tensor([1.0])))
    # print(sampler.sample(model=model, num_to_sample = 100, properties=torch.Tensor([[1], [0]]), max_seq_len = 100))

    #TODO: Test if max_seq_len is negotiable
    # sample = sampler.sample(model=model, properties=torch.Tensor([[1.0]]), num_to_sample=100, max_seq_len=100)
    # print(len(sample))
    # print(sample)

    # This strategy creates all the same molecule ... use_random set to true causes errors ...