import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

import os

import sys
import time
from datetime import timedelta
import json

# from sd_lstm_model import ConditionalSDLSTM
from model_sd_vae import SDVAE


def load_model(model_definition, model_weights, device, model_class=SDVAE,  copy_to_cpu=True):
    """

        Args:
            model_class: what class of model?
            model_definition: path to model json
            model_weights: path to model weights
            device: cuda or cpu
            copy_to_cpu: bool

        Returns: an RNN model

        """
    json_in = open(model_definition).read()
    raw_dict = json.loads(json_in)
    model = model_class(**raw_dict)
    map_location = lambda storage, loc: storage if copy_to_cpu else None
    model.load_state_dict(torch.load(model_weights, map_location))
    return model.to(device)



def get_tensor_dataset_alt(grammar_encodings, rule_masks, properties_array):
    """
    Adjusted function to include shifted masks for syntax-directed model training.
    """

    grammar_tensor = torch.from_numpy(grammar_encodings).float()
    masks_tensor = torch.from_numpy(rule_masks).float()
    props_tensor = torch.from_numpy(properties_array).float()

    inp_tensor = grammar_tensor[:, :-1]
    target_tensor = grammar_tensor[:, 1:]

    masks_tensor = masks_tensor[:, 1:] # Shift masks forward

    return TensorDataset(inp_tensor, target_tensor, masks_tensor, props_tensor)


def get_tensor_dataset(grammar_encodings, rule_masks, properties_array):
    """
    Adjusted function to include shifted masks for syntax-directed model training.
    """

    grammar_tensor = torch.from_numpy(grammar_encodings).float()
    masks_tensor = torch.from_numpy(rule_masks).float()
    props_tensor = torch.from_numpy(properties_array).float()

    inp_tensor = grammar_tensor[:, :-1]
    target_tensor = grammar_tensor[:, 1:]

    # TODO: I'm not sure weather or not the masks should be shifted here!!!! If they should
    # not be shifted, this should be : masks = masks_tensor[:, :-1]
    # My Findings:
    #   First mask is allways the same [1, 0, 0, \ldots, 0]
    #   First enc is allways the same [1, 0, 0, \ldots, 0] <-- start token
    #   Final X encs are all [0, \ldots, 0, 0, 1] <-- Some sort of sink state
    #   Final 
    # Example: train[0]
    # Encodings: [100....], Rule, Rule, Rule, Rule, [...001], [...001]
    # Masks    : [100....], Rule, Rule, Rule, Rule, [...001], [...001]
    # At rule k: the mask allows the rule for encoding k
    # My Reasoning:
    # - Therefore first enc is some initial state [start], the final ones represent no more possible ruls
    # - This means that mask K is saying what is allowed at step k, and enc k is saying what was selected at step k
    # So we should consider prediction at k-1 to predict logits for step k, and use mask k to determine what is allowed
    
    masks_tensor = masks_tensor[:, 1:]  # Masks in line with targets (makes sense siunce should mask out logits for comparison with targets) [alternative is masks_tensor[:, :-1]]

    return TensorDataset(inp_tensor, target_tensor, masks_tensor, props_tensor)



def load_encodings_masks_and_properties(data_path, 
                               properties_file = 'QM9_clean.csv', 
                               indecies_file = 'data_splits.npy',
                               encodings_file = 'grammar_encodings.npy',
                               masks_file = 'grammar_encoding_masks.npy',
                               rm_duplicates=False
                               ):
    '''
    Loads the data from datapath, removes invalid smiles (should be unneccesary) and removes dupliaces 

    Return: [property_list, train_x_enc, train_x_masks, train_y, valid_x_enc, valid_x_masks, valid_y]

    property_list is list of strings describing properties (e.g. ['LogP', 'QED'])
    train/valid enc and masks are dim (n_instances x max_applic x |rules|) [def: 13kish, 100, 80]
    train/valid y are dim (n_instances x n_properties) [def: 10k, 100, 80]
    '''
    try:
        # Load full dataset with properties
        dataset = pd.read_csv(os.path.join(data_path, properties_file))
        # Load encodings and masks
        grammar_encodings = np.load(os.path.join(data_path, encodings_file))
        grammar_masks = np.load(os.path.join(data_path, masks_file))
        # Load split indecies
        spl_indecies = np.load(os.path.join(data_path, indecies_file) )
    except:
        raise Exception(f"Could not load dataset, encodings, masks or indecies file from {data_path}. Have you pre-processed the dataset?")

    property_list = list(dataset.columns.drop(['QM9_id', 'SMILES']))

    train_x_enc = np.array(grammar_encodings[spl_indecies == 0])
    train_x_masks = np.array(grammar_masks[spl_indecies == 0])
    train_y = np.array(dataset[spl_indecies == 0].drop(['SMILES', 'QM9_id'], axis=1))
    
    valid_x_enc = np.array(grammar_encodings[spl_indecies == 1])
    valid_x_masks = np.array(grammar_masks[spl_indecies == 1])
    valid_y = np.array(dataset[spl_indecies == 1].drop(['SMILES', 'QM9_id'], axis=1))

    return property_list, train_x_enc, train_x_masks, train_y, valid_x_enc, valid_x_masks, valid_y


def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def save_model(model, base_dir, base_name):
    model_params = os.path.join(base_dir, base_name + '.pt')
    torch.save(model.state_dict(), model_params)

    model_config = os.path.join(base_dir, base_name + '.json')
    with open(model_config, 'w') as mc:
        mc.write(json.dumps(model.config))
