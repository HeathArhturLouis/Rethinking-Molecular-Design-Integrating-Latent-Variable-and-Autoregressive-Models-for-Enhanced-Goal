import json
import os
import time
from datetime import timedelta

import numpy as np
import torch

from guacamol.utils.data import remove_duplicates
from torch.utils.data import TensorDataset


from rnn_model import ConditionalSmilesRnn

import pandas as pd
import os

import sys

sys.path.append('../utils/')
from smiles_char_dict import SmilesCharDictionary


def get_tensor_dataset(smiles_array, properties_array):
    """
    Gets a numpy array of indices, convert it into a Torch tensor,
    divided it into inputs and targets and wrap it
    into a TensorDataset

    Args:
        numpy_array: to be converted

    Returns: a TensorDataset
    """

    tensor = torch.from_numpy(smiles_array).long()
    props = torch.from_numpy(properties_array).float()

    inp = tensor[:, :-1]
    target = tensor[:, 1:]

    return TensorDataset(inp, target, props)


def get_tensor_dataset_on_device(numpy_array, device):
    """
    Get tensor dataset and send it to a device
    Args:
        numpy_array: to be converted
        device: cuda | cpu

    Returns:
        a TensorDataset on the required device
    """

    dataset = get_tensor_dataset(numpy_array)
    dataset.tensors = tuple(t.to(device) for t in dataset.tensors)
    return dataset


def load_model(model_class, model_definition, model_weights, device, copy_to_cpu=True):
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


def load_rnn_model(model_definition, model_weights, device, copy_to_cpu=True):
    return load_model(ConditionalSmilesRnn, model_definition, model_weights, device, copy_to_cpu)


def save_model(model, base_dir, base_name):
    model_params = os.path.join(base_dir, base_name + '.pt')
    torch.save(model.state_dict(), model_params)

    model_config = os.path.join(base_dir, base_name + '.json')
    with open(model_config, 'w') as mc:
        mc.write(json.dumps(model.config))

def load_smiles_and_properties(data_path, 
                               dataset_file = 'QM9_clean.csv', 
                               indecies_file = 'data_splits.npy', 
                               fixed_len_numeric_file = 'QM9_fixed_numeric_smiles.csv',
                               rm_duplicates=False, 
                               max_len=100):
    '''
    Loads the data from datapath, removes invalid smiles (should be unneccesary) and removes dupliaces 

    Return: [train_x, train_y, valid_x, valid_y]
    train_x, valid_x are np.Array[string] <--- smiles strings
    train_y, valid_y are np.Array[np.Array[float]] <--- properties
    '''
    # TODO: Move this logic to trainer class
    if 'QM9' in data_path:
        dataset_file = 'QM9_clean.csv'
        indecies_file = 'data_splits.npy'
        fixed_len_numeric_file = 'QM9_fixed_numeric_smiles.csv'
        id_name = 'QM9_id'
    elif 'ZINC250K' in data_path:
        dataset_file = 'ZINC_clean.csv'
        indecies_file = 'data_splits.npy'
        fixed_len_numeric_file = 'ZINC_fixed_numeric_smiles.csv'
        id_name = 'ZINC_ID'
    else:
        raise Exception('Dataset not supported.')

    try:
        # Load full dataset with properties
        dataset = pd.read_csv(os.path.join(data_path, dataset_file))
        # Load fixed_len_numeric
        fixed_len_numeric = pd.read_csv(os.path.join(data_path, fixed_len_numeric_file))
        # Load split indecies
        spl_indecies = np.load( os.path.join(data_path, indecies_file) )
    except:
        raise Exception(f"Could not load dataset, numeric representation or indecies file from {data_path}. Have you pre-processed the dataset?")

    invalid_indecies = []

    property_list = list(dataset.columns.drop([id_name, 'SMILES']))

    #TODO: Convert to array?
    train_x = np.array(fixed_len_numeric[spl_indecies == 0])
    valid_x = np.array(fixed_len_numeric[spl_indecies == 1])

    train_y = np.array(dataset[spl_indecies == 0].drop(['SMILES', id_name], axis=1))
    valid_y = np.array(dataset[spl_indecies == 1].drop(['SMILES', id_name], axis=1))

    return property_list, train_x, train_y, valid_x, valid_y


def rnn_start_token_vector(batch_size, device='cpu'):
    """
    Returns a vector of start tokens for SmilesRnn.
    This vector can be used to start sampling a batch of SMILES strings.

    Args:
        batch_size: how many SMILES will be generated at the same time in SmilesRnn
        device: cpu | cuda

    Returns:
        a tensor (batch_size x 1) containing the start token
    """
    sd = SmilesCharDictionary()
    return torch.LongTensor(batch_size, 1).fill_(sd.begin_idx).to(device)


def time_since(start_time):
    seconds = int(time.time() - start_time)
    return str(timedelta(seconds=seconds))


def set_random_seed(seed, device):
    """
    Set the random seed for Numpy and PyTorch operations
    Args:
        seed: seed for the random number generators
        device: "cpu" or "cuda"
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
