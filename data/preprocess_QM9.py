import glob
import time
import random
import sys
import argparse
import os

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from tqdm import tqdm


sys.path.append('./utils/')
from smiles_char_dict import SmilesCharDictionary
from property_calculator import PropertyCalculator

import cfg_parser

from config import CONFIG

from mol_tree import AnnotatedTree2MolTree
from tree_walker import OnehotBuilder
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks

# from mol_util import DECISION_DIM
def syntax_encode_smiles(smiles_list):
    '''
    Inputs: 
        smiles_list -> List of SMILES strings
    Outputs:
        onehot -> (len(smiles_list), max_decode_steps, decision_dim)
        masks -> (len(smiles_list), max_decode_steps, decision_dim)

    Outputs lists of rules to apply at each decoding step and mask specifying what rules are valid at that step
    - decision_dim is possible rules to apply
    - max_decode_steps is maximum number of rules that can be applied to gen a mol
    '''
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


def generate_data_splits(target_path, n_molecules, test_size, validation_size, random_state=None):
    '''
    Generate test val train splits for QM9 dataset and save them in target dir
    '''
    if random_state is None:
        random.seed(time.time())
    else:
        random.seed(random_state)

    indices = np.arange(n_molecules)
    random.shuffle(indices)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+validation_size]
    # Not needed
    # train_indices = indices[test_size+validation_size:]

    splits = np.zeros(n_molecules)
    splits[test_indices] = 2  # Test set
    splits[val_indices] = 1  # Validation set
    # train set remains zero

    np.save(target_path + '/data_splits.npy', splits)


def preprocess_qm9(raw_data_dir, target_path, prop_names = False, verbose = False, check_validity = True, remove_duplicates = True, generate_numeric = True, max_len = 100, clean_heavy_atoms = True):
        '''
        Preprocess and clean QM9 dataset, compute molecular properties

        csv format:
        QM9_id, SMILES string, Fixed length integer array representation of SMILES, QED, SA, LogP

        Returns: Number of molecules in dataset
        '''

        
        # TODO: Make cross platform
        files = sorted(glob.glob(raw_data_dir + '/*'))
    
        n_valid = 0

        # Write header
        dataset = pd.DataFrame(columns=['QM9_id','SMILES'] + CONFIG.target_props)
        # + 2 to account for sd.BEGIN and sd.END
        fixed_numeric_ds = pd.DataFrame(columns=[('Pos' + str(a)) for a in range(max_len + 2)])

        # print('QM9_id,SMILES,FIXED_LEN_NUMERIC_SMILES,QED,SA,LogP\n', file=ofile)

        n_invalid = 0

        if (not prop_names):
            pc = PropertyCalculator(CONFIG.target_props)
        else:
            pc = PropertyCalculator(prop_names)
        

        grammar_enc = []
        grammar_enc_masks = []

        # For each molecule record in target dir
        for f in tqdm(files, desc='Processing Molecules'):
            sd = SmilesCharDictionary()

            #if verbose:
            #    print(f)
            lines = open(f, "r").readlines()
            raw = lines[-2].split('\t')[0]
            mol = Chem.MolFromSmiles(raw)
            smiles = Chem.MolToSmiles(mol)
            # Clean invalid SMILES

            if smiles is None or (check_validity and not (sd.allowed(smiles) and len(smiles) < max_len)):
                # If we're checking validity and SMILES is either too long of contains symbols dissalowed by the dictionary, ignore it :
                n_invalid += 1
                continue
            else:
                n_valid += 1

            mol = Chem.MolFromSmiles(smiles)
            # Compute properties
            #logP = Descriptors.MolLogP(mol)
            #sa_score = sascorer.calculateScore(mol)
            #qed = QED.qed(mol)

            properties = pc(mol)
            # Check validity

            smiles = smiles.strip()
            smiles = sd.encode(smiles)

            # Get numeric representation
            # TODO: Minor optimizaiton would be to use min or max_len and largest smile still in the dataset
            numeric_rep = np.zeros(max_len + 2, dtype=np.int32)

            # Add numeric rep + beginning and end tokens
            numeric_rep[0] = sd.char_idx[sd.BEGIN]
            for i in range(len(smiles)):
                numeric_rep[i+1] = sd.char_idx[smiles[i]]

            numeric_rep[len(smiles)+1] = sd.char_idx[sd.END]

            dataset.loc[n_valid] = [f.split('_')[1][:-4], smiles] + properties
            fixed_numeric_ds.loc[n_valid] = numeric_rep
            # print(','.join([str(a) for a in (f.split('_')[1][:-4], smiles, numeric_rep, qed, sa_score, logP)]), file=ofile)

            # Compute grammar encoding and grammar encoding mask
            
            ge, gem = syntax_encode_smiles([smiles])

            grammar_enc.append(ge)
            grammar_enc_masks.append(gem)

        # Remove Duplicates
        if remove_duplicates:
            print('Removing Duplicates')
            duplicates = dataset.duplicated(keep='first')
            duplicate_indices = np.where(duplicates)[0]
            dataset.drop(duplicate_indices, inplace=True)
            fixed_numeric_ds.drop(duplicate_indices, inplace=True)

            print(f'    Removed {len(duplicate_indices)} duplicated rows.')
    
        print('Saving Data')
        dataset.to_csv(os.path.join(target_path, 'QM9_clean.csv'), header=True, index=False)
        fixed_numeric_ds.to_csv( os.path.join(target_path, 'QM9_fixed_numeric_smiles.csv'), header=True, index=False)
        # Write grammar encoding and grammar encoding masks
        np.save(os.path.join(target_path, 'grammar_encodings.npy'), np.array(grammar_enc).squeeze(axis=1))
        np.save(os.path.join(target_path, 'grammar_encoding_masks.npy'), np.array(grammar_enc_masks).squeeze(axis=1))

        if verbose:
            print( f'   {n_valid} smiles kept and {n_invalid} smiles removed [out of total {n_valid + n_invalid}]')
        return n_valid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example script with default values')
    ### QM9 Arguments
    parser.add_argument('--dset_name', type=str, default='QM9', help='Dataset to parse')
    parser.add_argument('--QM9_raw_data_path', type=str, default='./data/QM9/dsgdb9nsd.xyz/', help='Path to dir containing raw QM9 data.')
    parser.add_argument('--QM9_target_path', type=str, default='./data/QM9/QM9_36/', help='Path to dir to store QM9 csv and split indecies.')
    parser.add_argument('--QM9_max_smile_len', type=int, default=36, help='SMILES longer than this will be discarded.')
    parser.add_argument('--QM9_prop_names', type=list, default=[], help='List of properties to overrride the one found in config')
    ### QM9 Split Sizes
    parser.add_argument('--QM9_test_size', type=int, default=10000, help='Number of mols in train set. (remainder will be used for train)')
    parser.add_argument('--QM9_validation_size', type=int, default=10000, help='Number of mols in validation set. (remainder will be used for train)')
    parser.add_argument('--QM9_split_seed', type=int, default=42, help='Seed for determining random splits')
    ### MISC
    parser.add_argument('--verbose', type=bool, default=True, help='Path to dir to store QM9 csv and split indecies.')

    args = parser.parse_args()

    if(args.dset_name == 'QM9'):
        print('Cleaning QM9 Data and Generating Properties.')
        n_valid = preprocess_qm9(raw_data_dir=args.QM9_raw_data_path,
                       target_path=args.QM9_target_path,
                       verbose=args.verbose,
                       prop_names=args.QM9_prop_names,
                       max_len = 36,# 34,
                       clean_heavy_atoms = True)

        print('Generating Test-Train-Validation Splits.')
        generate_data_splits(target_path=args.QM9_target_path, 
                             n_molecules=n_valid, 
                             test_size=args.QM9_test_size,
                             validation_size=args.QM9_validation_size, 
                             random_state=args.QM9_split_seed)



