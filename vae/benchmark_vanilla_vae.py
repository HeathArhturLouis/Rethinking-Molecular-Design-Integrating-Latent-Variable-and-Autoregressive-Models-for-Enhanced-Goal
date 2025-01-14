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
    def __init__(self, batch_size, device, sampling_std_div = 1.0):
        self.batch_size = batch_size
        self.device = device
        self.sd = SmilesCharDictionary()

        self.sampling_std_div = sampling_std_div

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

            # Probably redundant if model.reparam flag is set to false, remove in final version of code?
            z = model.reparameterize(mu = z, logvar = torch.tensor(model.eps_std))

            out_logits = model.state_decoder(z, normd_props).permute(1, 0, 2)

        if not random:
            out_tokens = torch.argmax(out_logits, dim=-1)
        else:
            probs = torch.softmax(out_logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            out_tokens = distribution.sample()  

        # out tokens is shape seq_len x batch_size

        out_smiles = self.sd.matrix_to_smiles(out_tokens)
        
        # Return Accuracy / Junk

        return out_smiles
        # return self.cal_accuracy(out_smiles, input_smiles)

    # Unconditional Sampling for Unconditional Performance

    def _sample_prior_unbatched(self, model, properties, random, latent_points = None):
        '''
        Sample random points in the prior with given properties n_to_sample times
        This is the strategy applied in the SD VAE code (fixed test props and random sampled z's)

        if z != None fixes latent points

        resets value of reparam after being called
        '''

        old_reparam = model.reparam

        normd_props = model.normalize_prop_scores(properties)

        n_to_sample = normd_props.shape[0]
        # Sample latent points

        if latent_points is None:
            latent_points = np.random.normal(0, self.sampling_std_div, size=(n_to_sample, model.latent_dim))
            # latent_points = np.random.normal(0, model.eps_std, size=(n_to_sample, model.latent_dim))

        latent_points = torch.tensor(latent_points, dtype=torch.float32)
        # Latent Dist is shape n_to_sample, hidden_dim
        
        # Reparam can be false since we're already sampling from the prior / have already provided latent points sampled from the prior
        model.reparam = False
        model.eval()

        with torch.no_grad():
            raw_logits = model.state_decoder(latent_points, normd_props).permute(1, 0, 2)

        # Raw logits is shape n_to_sample x max_seq_len x decision_dim
        if not random:
            tokens = torch.argmax(raw_logits, dim=-1)

        else:
            probs = torch.softmax(raw_logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            tokens = distribution.sample()

        out_smiles = self.sd.matrix_to_smiles(tokens)

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

'''
def cal_valid_prior(model, latent_dim, labels, nb_latent_point, sample_times, chunk_size, sigma):
    import rdkit
    from rdkit import Chem
    whole_valid, whole_total = 0, 0
    valid_smile = []
    pbar = tqdm(list(range(0, nb_latent_point, chunk_size)), desc='decoding')
    for start in pbar:
        end = min(start + chunk_size, nb_latent_point)
        # SAMPLE LATENT POINT
        latent_point = np.random.normal(0, sigma, size=(end - start, latent_dim))
        latent_point = latent_point.astype(np.float32)
        #y = np.tile(labels, (nb_latent_point,1)) 
        
        # GET LABELS PROVIDED AS ARGUMENT SOMEHOW
        y = labels[:end-start].astype(np.float32)

        # GET LOGITS AND DECODE TO SMILES
        raw_logits = model.pred_raw_logits(latent_point, y)
        decoded_array = batch_decode(raw_logits, True, decode_times=sample_times)

        for i in range(end - start):
            for j in range(sample_times):
                s = decoded_array[i][j]
                if not s.startswith('JUNK') and Chem.MolFromSmiles(s) is not None:
                    whole_valid += 1
                    valid_smile.append(s)
                whole_total += 1
        pbar.set_description('valid : total = %d : %d = %.5f' % (whole_valid, whole_total, whole_valid * 1.0 / whole_total))
    return 1.0 * whole_valid / whole_total, whole_valid, whole_total, valid_smile
'''

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

def benchmark_reconstruction_QM9(model, sampler):
    # Load test SMILES from data_path

    data_splits = np.load('../data/QM9/data_splits.npy')
    # Load test PROPERTIES
    all_QM9 = pd.read_csv('../data/QM9/QM9_clean.csv')
    test_props = np.array((all_QM9['LogP']))[data_splits == 2]
    test_smiles = np.array((all_QM9['SMILES']))[data_splits == 2]

    test_props = torch.tensor([[a] for a in test_props])

    recon_smiles = sampler.reconstruct_smiles(model, test_smiles, test_props)

    assert len(recon_smiles) == len(test_smiles)

    same = 0
    junk = 0
    for i in range(len(recon_smiles)):
        if recon_smiles[i] == test_smiles[i]:
            same += 1
        if 'JUNK' in recon_smiles[i]:
            junk += 1
    
    acc = same / len(recon_smiles)
    junk_pct = junk / len(recon_smiles)
    print(f'Accuracy: { acc }')
    print(f'Junk PCT: { junk_pct }')


if __name__ == '__main__':
    model_weights = '../models/REG_VAE_BEST/SD_REG_VANILLA_VAE_mute-brook-70_Epoch_261_Vl_0.161.pt'
    model_definit = '../models/REG_VAE_BEST/SD_REG_VANILLA_VAE_mute-brook-70_Epoch_261_Vl_0.161.json'

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