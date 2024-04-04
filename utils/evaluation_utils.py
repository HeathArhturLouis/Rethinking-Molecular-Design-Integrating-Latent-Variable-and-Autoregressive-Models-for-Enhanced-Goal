from rdkit import Chem

import sys
sys.path.append('../utils/')
from property_calculator import PropertyCalculator

import numpy as np
from tqdm import tqdm

from rdkit.Chem import Draw

import matplotlib.pyplot as plt

from joblib import Parallel, delayed



def return_valid_smiles(smiles_list):
    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        try:
            # Translating back from rdkit yields the canonical representation
            valid_smiles.append(Chem.MolToSmiles(mol))
        except:
            pass

    return valid_smiles



def plot_smiles(smiles_list, title="SMILES Plot"):
    """
    Plots a 10x10 grid of molecular structures from a list of SMILES strings,
    with no space between plots and no axes.

    Args:
    smiles_list (list): List of SMILES strings.
    title (str): Title of the plot.
    """
    # remove invalid smiles and make sure we have enough
    smiles_list = return_valid_smiles(smiles_list)
    assert len(smiles_list) > 100

    # pick 100 at random
    smiles_list = np.random.choice(smiles_list, size=100, replace=False)
    
    # Convert SMILES to RDKit molecule objects
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list]

    # Create a 10x10 subplot with no spaces between subplots
    fig, axs = plt.subplots(10, 10, figsize=(10, 10))
    fig.suptitle(title, fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)  # Remove spaces between plots

    for i, ax in enumerate(axs.flat):
        ax.axis("off")  # Turn off the axes

        if i < len(molecules):
            mol = molecules[i]
            if mol is not None:
                # Convert RDKit molecule to an image and display it
                img = Draw.MolToImage(mol, size=(150, 150))
                ax.imshow(img)

    plt.show()

def props_from_smiles(smiles_list, verbose=True, prop_names = ['LogP']):
    '''
    Computes property scores of all valid smiles and returns as list
    '''
        
    pc = PropertyCalculator(prop_names)
    if verbose:
        smiles_list = tqdm(smiles_list)
    props = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            props.append(pc(mol)[0])
    return props


def amina_metrics(smiles_list, train_smiles):
    '''
    Compute validity, novelty and uniqueness in the same manner as SD-VAE code base

    validity:  # valid generated molecules (accepted by RDKIT) / # all generations
    uniqueness:  # unique can. valid generated molecules / # can. valid generated molecules

    # Following the guacamol definition since the computation in the old code base isn't up to date

    'The novelty score is calculated by generatingmolecules, until 10 000 different canonical SMILES strings areobtained, 
    and computing the ratio of molecules not present inthe ChEMBL data set.'
    
    novelty:  # valid can. uni. molecules not in training / # all uni. can. valid generated molecules


    return [validity, uniqueness, novelty]
    '''

    valid_smiles = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        try:
            # Translating back from rdkit yields the canonical representation
            valid_smiles.append(Chem.MolToSmiles(mol))
        except:
            pass


    unique = set(valid_smiles)

    validity = len(valid_smiles) / len(smiles_list)
    uniqueness = len(unique) / len(valid_smiles)

    intersection = unique & set(train_smiles)

    # Careful here! If dataset is not cacnonicalized it should be for comparison
    novelty = (len(unique) - len(intersection)) / len(unique)

    return validity, uniqueness, novelty


def property_metrics(smiles_list, target_props, prop_names=['LogP']):
    '''
    Compute property correlation and MSE
    ignore invalid molecules
    '''
        
    
    pc = PropertyCalculator(prop_names)

    assert len(smiles_list) == len(target_props)

    valid_target_props = []
    valid_computed_props = []

    for smiles, tprop in zip(smiles_list, target_props):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            computed_props = pc(mol)
            if computed_props: 
                valid_target_props.append(tprop)
                valid_computed_props.append(computed_props)

    maes = []
    corrs = []

    for i in range(len(prop_names)):
        try:

            computed = [prop[i] for prop in valid_computed_props]
            target = [tprop[i] for tprop in valid_target_props]
            squared_diff = [abs((a - b)) for a, b in zip(computed, target)]
            mse = sum(squared_diff) / len(squared_diff)
            maes.append(mse)

            corrs.append(np.corrcoef(computed, target)[0, 1])
        except Exception as e:
            corrs.append(f'ERROR: Cannot compute correlation')

    return corrs, maes


def reconstruct_single(model, smiles, labels, encode_times, decode_times):
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encoder(chunk)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1, labels)
            print(_result)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result


def reconstruct(model, smiles, labels, chunk_size, encode_times, decode_times):
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size], labels[chunk_start: chunk_start + chunk_size], encode_times, decode_times)
        for chunk_start in range(0, len(smiles), chunk_size)
    )

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result


def reconstruct_normalize(model, smiles, labels, chunk_size, encode_times, decode_times):
    labels = model.normalize_prop_scores(labels)
    return reconstruct(model, smiles, labels, chunk_size, encode_times, decode_times)




if __name__ == '__main__':
    print(property_metrics(smiles_list = ['CCO', 'CCCC', 'ERRRORORORORO', 'COO'], target_props=[[-0.001400000000],[1.8064], [-3.2223] ,[0.1058000]]))