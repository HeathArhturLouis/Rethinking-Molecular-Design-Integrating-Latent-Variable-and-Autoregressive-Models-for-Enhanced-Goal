import torch
import numpy as np
import math

# from typing import Type

import torch.nn.functional as F
from torch import nn

import sys
import os

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)) , '../utils'))
from rnn_utils import load_model
from model_vanilla_vae import VanillaMolVAE

from smiles_char_dict import SmilesCharDictionary


class VanillaVAEHarness:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.sd = SmilesCharDictionary()

    def cal_accuracy(self, decode_result, smiles):
        accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
        junk = [sum([1 for c in cand if c.startswith('JUNK')]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
        return (sum(accuracy) * 1.0 / len(accuracy)), (sum(junk) * 1.0 / len(accuracy))
    
    def reconstruct_smiles(self, model, input_smiles, target_props, random=False):
        num_batches = len(input_smiles) // self.batch_size + (len(input_smiles) % self.batch_size != 0)
        
        out_smiles = []
        il = 0
        iu = self.batch_size

        for i in tqdm(range(num_batches)):
            batch_smiles = self._reconstruct_smiles_unbatched(model, input_smiles[il:iu], target_props[il:iu, :], random=random)
            out_smiles += batch_smiles

            il += self.batch_size
            iu = min(iu + self.batch_size, len(input_smiles))

        return out_smiles

    def _reconstruct_smiles_unbatched(self, model, input_smiles, target_props, random):
        '''
        Returns reconstruction of all input tokens

        Does not reparameterise,

        model : vanilla VAE model
        input_smiles : batch_size x max_seq_len array of tokens representing a smiles
        target_props : 2D unnormalized tensor of properties batch_size x prop_size
        '''
        # Reparameterization is generally not neccesary during inference
        # This flag works as it is checked inside the reparam function
        model.reparam = False
        model.eval()

        if not torch.is_tensor(target_props):
            target_props = torch.tensor(target_props)
    
        assert len(input_smiles) == target_props.shape[0]
        assert len(model.property_names) == target_props.shape[1]

        # TODO: Assert max seq len is correct for model
        max_seq_len = model.max_decode_steps
        b_size = len(input_smiles)

        # Convert smiles to tokens
        tokens = torch.zeros([b_size, max_seq_len], dtype=torch.long)

        for i in range(len(input_smiles)):
            smi = input_smiles[i]
            tokens[i][0] = self.sd.char_idx[self.sd.BEGIN]

            for j in range(len(smi)):
                tokens[i][j+1] = self.sd.char_idx[smi[j]]
            
            tokens[i][len(smi)+1] = self.sd.char_idx[self.sd.END]

        # Normalize properties
        normd_props = model.normalize_prop_scores(target_props)

        with torch.no_grad():
            # Compute hidden state treating entire input as atch
            z = model.encoder(tokens)[0]

            # redundant if model.reparam flag is set to false, remove in final version of code?
            z = model.reparameterize(mu = z, logvar = torch.tensor(model.eps_std))

            # out_logits = model.state_decoder(z, normd_props, return_logits = True).permute(1, 0, 2)
            # out_actions = model.state_decoder(z, normd_props, return_logits = False)
            # raise Exception('Reconstruction: state_decoder is randomly sampling actions')
            # TODO: Check weather the matrix is interpreted with argmax or sampling
            out_actions = model.state_decoder(z, normd_props, x_inputs = None, teacher_forcing = False, return_logits = False)

            # out_tokens = torch.argmax(out_logits, dim=-1)
            
        # out tokens is shape seq_len x batch_size

        # Remove start token
        out_smiles = self.sd.matrix_to_smiles(out_actions[:, 1:])
        
        # Return Accuracy / 

        return out_smiles
        # return self.cal_accuracy(out_smiles, input_smiles)

    # Unconditional Sampling for Unconditional Performance

    def _sample_prior_unbatched(self, model, properties, random, latent_points=None):
        '''
        Sample random points in the prior with given properties n_to_sample times
        This is the strategy applied in the SD VAE code (fixed test props and random sampled z's)
        '''

        old_reparam = model.reparam

        normd_props = model.normalize_prop_scores(properties)

        n_to_sample = normd_props.shape[0]
        # Sample latent points
        if latent_points is None:
            # latent_points = np.random.normal(0, model.eps_std, size=(n_to_sample, model.latent_dim))
            latent_points = np.random.normal(0, 1.0 , size=(n_to_sample, model.latent_dim))
        
        latent_points = torch.tensor(latent_points, dtype=torch.float32)
        # Latent Dist is shape n_to_sample, hidden_dim
        
        # Sample logits
        model.reparam = False
        model.eval()

        with torch.no_grad():
            # raw_logits = model.state_decoder(latent_points, normd_props)[0].permute(1, 0, 2)
            out_actions = model.state_decoder(latent_points, normd_props, x_inputs = None, teacher_forcing = False, return_logits = False)[:, 1:]

        out_smiles = self.sd.matrix_to_smiles(out_actions)

        model.reparam = old_reparam

        return out_smiles
    
    def sample(self, model, properties, random=True, latent_points = None):
        # Max decode steps is fixed for VAE
        # assert max_seq_len == model.max_decode_steps

        max_seq_len = model.max_decode_steps

        properties = torch.tensor(properties).clone()

        num_to_sample = properties.shape[0]

        assert properties.shape[1]

        num_batches = num_to_sample // self.batch_size + (num_to_sample % self.batch_size != 0)
        
        out_smiles = []
        il = 0
        iu = self.batch_size

        for i in range(num_batches):
 
            if latent_points is not None:
                batch_latent = latent_points[il:iu, :]
            else:
                batch_latent = None

            batch_smiles = self._sample_prior_unbatched(model, properties[il:iu, :], random=random, latent_points=batch_latent)
            out_smiles += batch_smiles

            il += self.batch_size
            iu = min(iu + self.batch_size, num_to_sample)

        return out_smiles

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


import pandas as pd

if __name__ == '__main__':
    # model_weights = '../models/REG_VAE_BEST/SD_REG_VANILLA_VAE_mute-brook-70_Epoch_261_Vl_0.161.pt'
    # model_definit = '../models/REG_VAE_BEST/SD_REG_VANILLA_VAE_mute-brook-70_Epoch_261_Vl_0.161.json'

    # model_weights = '../models/EXPLICIT_VANILLA_VAE/SD_LSTM_throbbing-unit-04_Epoch_99_Vl_0.440.pt'
    # model_definit = '../models/EXPLICIT_VANILLA_VAE/SD_LSTM_throbbing-unit-04_Epoch_99_Vl_0.440.json'

    model_weights = '../models/EXPLICIT_VANILLA_VAE_TF_03/SD_LSTM_odd-sunset-14_Epoch_31_Vl_0.351.pt'
    model_definit = '../models/EXPLICIT_VANILLA_VAE_TF_03/SD_LSTM_odd-sunset-14_Epoch_31_Vl_0.351.json'

    sampler = VanillaVAEHarness(batch_size=64, device='cpu')
    model = load_model(model_class=VanillaMolVAE, model_definition=model_definit, model_weights=model_weights, device='cpu')

    benchmark_reconstruction_QM9(model, sampler)

    # properties = torch.tensor([[1.122323] for _ in range(100)], dtype=torch.float32)
    
    # new_smiles = sampler.sample_prior_unbatched(model, 100, properties)
    # new_smiles = sampler.sample(model, properties, num_to_sample=100, max_seq_len=101)
    # print(new_smiles)

    '''
    out_smiles = sampler.reconstruct_smiles(model, test_smiles, target_props)
    print(out_smiles)
    print(test_smiles)
    accurate, junk = sampler.cal_accuracy(out_smiles, test_smiles)
    print(f'Accuracy: {accurate}')
    print(f'Junk: {junk}')
    '''