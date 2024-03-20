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
from model_sd_vae import SDVAE

import cfg_parser
from mol_tree import AnnotatedTree2MolTree
from tree_walker import OnehotBuilder
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks

from sd_smiles_decoder import raw_logit_to_smiles

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))

GRAMMAR_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../grammar/mol_zinc.grammar')


class VanillaVAEHarness:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.grammar = cfg_parser.Grammar(GRAMMAR_PATH)

    def cal_accuracy(self, decode_result, smiles):
        accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
        junk = [sum([1 for c in cand if c.startswith('JUNK')]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
        return (sum(accuracy) * 1.0 / len(accuracy)), (sum(junk) * 1.0 / len(accuracy))
    
    def reconstruct_smiles(self, model, input_smiles, target_props):
        num_batches = len(input_smiles) // self.batch_size + (len(input_smiles) % self.batch_size != 0)
        
        out_smiles = []
        il = 0
        iu = self.batch_size

        for i in tqdm(range(num_batches)):
            batch_smiles = self._reconstruct_smiles_unbatched(model, input_smiles[il:iu], target_props[il:iu, :])
            out_smiles += batch_smiles

            il += self.batch_size
            iu = min(iu + self.batch_size, len(input_smiles))

        return out_smiles

    def _reconstruct_smiles_unbatched(self, model, input_smiles, target_props):
        '''
        Returns reconstruction of all input tokens

        model : vanilla VAE model
        input_smiles : batch_size x max_seq_len array of tokens representing a smiles
        target_props : 2D unnormalized tensor of properties batch_size x prop_size
        '''
        model.reparam = False
        model.eval()

        if not torch.is_tensor(target_props):
            target_props = torch.tensor(target_props)
    
        assert len(input_smiles) == target_props.shape[0]
        assert len(model.property_names) == target_props.shape[1]

        # TODO: Assert max seq len is correct for model
        max_seq_len = model.max_decode_steps
        b_size = len(input_smiles)

        # Normalize properties
        normd_props = model.normalize_prop_scores(target_props)

        # Convert smiles to tokens and masks
        cfg_tree_list = []

        for smiles in input_smiles:
            ts = cfg_parser.parse(smiles, self.grammar)
            assert isinstance(ts, list) and len(ts) == 1

            n = AnnotatedTree2MolTree(ts[0])
            cfg_tree_list.append(n)

        walker = OnehotBuilder()
        tree_decoder = create_tree_decoder()
        tokens, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

        # onehot is batch_size x max_seq_len x decision_dim
        # Prune tensor to max_seq_len ... TODO: Add option to extend?
        # TODO: Can this be avoided somehow?
        
        tokens = tokens[:, :model.max_decode_steps, :]

        # Reconstruct
        with torch.no_grad():
            # Compute hidden state treating entire input as atch
            tokens = torch.tensor(tokens, dtype=torch.float32).permute(0, 2, 1)
            z = model.encoder(tokens)[0]

            out_logits = model.state_decoder(z, target_props)

        # out tokens is shape seq_len x batch_size

        # out_smiles = self.sd.matrix_to_smiles(out_tokens)
        out_smiles = raw_logit_to_smiles(out_logits, use_random = False) # Sample most probable at each step
        # Return Accuracy / Junk

        return out_smiles
        # return self.cal_accuracy(out_smiles, input_smiles)

    def _sample_prior_unbatched(self, model, properties, use_random=True):
        '''
        Sample random points in the prior with given properties n_to_sample times
        This is the strategy applied in the SD VAE code (fixed test props and random sampled z's)
        '''
        n_to_sample = properties.shape[0]
        # Sample latent points
        latent_points = np.random.normal(0, model.eps_std, size=(n_to_sample, model.latent_dim))
        latent_points = torch.tensor(latent_points, dtype=torch.float32)
        # Latent Dist is shape n_to_sample, hidden_dim
        
        # Sample logits
        model.reparam = False
        model.eval()

        with torch.no_grad():
            raw_logits = np.array(model.state_decoder(latent_points, properties)) # .permute(1, 0, 2)

        # Raw logits is size max_seq_len x batch_size x decision dim
        out_smiles = raw_logit_to_smiles(raw_logits, use_random = use_random)
        
        return out_smiles
        
    def sample(self, model, properties, num_to_sample, max_seq_len, use_random=True):
        # Max decode steps is fixed for VAE
        assert max_seq_len == model.max_decode_steps
        # I really need to fix this interface
        assert properties.shape[0] == num_to_sample

        num_batches = num_to_sample // self.batch_size + (num_to_sample % self.batch_size != 0)
        
        out_smiles = []
        il = 0
        iu = self.batch_size

        for i in tqdm(range(num_batches)):
            batch_smiles = self._sample_prior_unbatched(model, properties[il:iu, :], use_random=use_random)
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
    model_weights = '../models/SD_VAE_MASKED_BIN_CE/SD_LSTM_lively-smoke-48_Epoch_12_Vl_0.134.pt'
    model_definit = '../models/SD_VAE_MASKED_BIN_CE/SD_LSTM_lively-smoke-48_Epoch_12_Vl_0.134.json'

    sampler = VanillaVAEHarness(batch_size=64, device='cpu')
    model = load_model(model_class=SDVAE, model_definition=model_definit, model_weights=model_weights, device='cpu')


    # benchmark_reconstruction_QM9(model, sampler)


    properties = torch.tensor([[1.0] for _ in range(100)], dtype=torch.float32)
    
    # new_smiles = sampler.sample_prior_unbatched(model, 100, properties)
    new_smiles = sampler.sample(model, properties, num_to_sample=100, max_seq_len=99)
    print(new_smiles)