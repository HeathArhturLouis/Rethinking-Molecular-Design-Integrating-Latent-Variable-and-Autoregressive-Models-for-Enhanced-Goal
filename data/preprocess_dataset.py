"""
Helper functions and utilities for dataset preprocessing.
"""

import glob
import time
import random
import sys
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from typing import List, Union, Tuple

sys.path.append("../utils/")
sys.path.append("../")

from smiles_char_dict import SmilesCharDictionary
from property_calculator import PropertyCalculator
import cfg_parser
from config import CONFIG
from mol_tree import AnnotatedTree2MolTree
from tree_walker import OnehotBuilder
from attribute_tree_decoder import create_tree_decoder
from batch_make_att_masks import batch_make_att_masks

# Define combined list like type for type checking
ListLike = Union[List, np.ndarray]


def syntax_encode_smiles(smiles_list: ListLike) -> Tuple[ListLike, ListLike]:
    """
    Outputs lists of rules to apply at each decoding step and mask specifying what rules are valid at that step.
    Here decision_dim is possible rules to apply and max_decode_steps is maximum number of rules that can be applied
    to generate a molecule.

    Args:
        smiles_list: List of SMILES strings
    Returns:
        [
            onehot: (len(smiles_list), max_decode_steps, decision_dim)
            masks: (len(smiles_list), max_decode_steps, decision_dim)
        ]
    """

    grammar = cfg_parser.Grammar(os.path.join("../", CONFIG.grammar_file))

    cfg_tree_list = []
    for smiles in smiles_list:
        ts = cfg_parser.parse(smiles, grammar)
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
    onehot, masks = batch_make_att_masks(
        cfg_tree_list, tree_decoder, walker, dtype=np.byte
    )

    return (onehot, masks)


def generate_data_splits(
    target_path: str,
    n_molecules: int,
    test_size: int,
    validation_size: int,
    random_state: int = None,
):
    """
    Generate test val train splits for QM9 dataset and save them as a nympy array of codes
    in directory specified by target path (0 for train, 1 for validation and 2 for testing).

    Args:
        target_path: String specifying an existing directory where to save the data split files.
        n_molecules: Number of total valid molecules in the dataset.
        test_size: Number of molecules to put in the test set.
        validation_size: Number of molecules to put in the validation set.
        random_state: Seed for random seed.
    Returns:
        None
    """
    assert validation_size + test_size < n_molecules

    if random_state is None:
        random.seed(time.time())
    else:
        random.seed(random_state)

    # Shuffle indecies as datasets like QM9 are often ordered by for example SMILES length
    indices = np.arange(n_molecules)
    random.shuffle(indices)

    # Sample random indices for test and train sets
    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + validation_size]

    # Assign codes
    splits = np.zeros(n_molecules)
    splits[test_indices] = 2
    splits[val_indices] = 1

    np.save(target_path + "/data_splits.npy", splits)


def preprocess_qm9(
    raw_data_dir: str,
    target_path: str,
    prop_names: List[str] = False,
    verbose: bool = False,
    check_validity: bool = True,
    remove_duplicates: bool = True,
    max_len: int = 100,
):
    """
    Wrapper function encapsulating QM9 preprocessing logic with flags for each data cleaning step:
        - Generates and saves CSV of SMILES and desired properties
        - Generates and saves numeric encodings as numpy array
        - Generates and saves grammar encodings (for use with syntax directed models)

    CSV format:
    QM9_id, SMILES string, Fixed length integer array representation of SMILES, QED, SA, LogP

    Args:
        raw_data_dir: Path to directory containing raw xyz files for QM9 molecules.
        target_path: Path to target directory to store cleaned dataset in.
        prop_names: List of property names to compute, must be supported by property calculator.
        verbose: Flags weather to print more detailed output for debugging.
        check_validity: Weather to remove invalid smiles when encountered.
        remove_duplicates: Weather to check and remove duplicate smiles.
        max_len: Determines padding length and maximum allowed length of SMILES in the cleaned dataset.
    Returns:
        int: Number of molecules in dataset
    """

    # TODO: Make this work properly cross platform and realy on a more compact and generic input format than xyz files
    files = sorted(glob.glob(raw_data_dir + "/*"))

    # Create empty dataframe with appropriate header
    dataset = pd.DataFrame(columns=["QM9_id", "SMILES"] + CONFIG.target_props)
    fixed_numeric_ds = pd.DataFrame(
        columns=[("Pos" + str(a)) for a in range(max_len + 2)]
    )  # + 2 to account for sd.BEGIN and sd.END

    # Track number of molecules cleaned and grammar encodings + masks
    n_valid = 0
    n_invalid = 0
    grammar_enc = []
    grammar_enc_masks = []

    # Initialize property calculator and character dictionary objects
    if not prop_names:
        pc = PropertyCalculator(CONFIG.target_props)
    else:
        pc = PropertyCalculator(prop_names)

    sd = SmilesCharDictionary()

    # For each molecule record in target dir
    for f in tqdm(files, desc="Processing Molecules"):
        # Parse the record into a RdKit molecule and canonical SMILES
        # TODO: Appears to not work for all possible valid xyz formats. Look into .xyz and make generic.
        lines = open(f, "r").readlines()
        raw = lines[-2].split("\t")[0]

        try:
            mol = Chem.MolFromSmiles(raw)
            smiles = Chem.MolToSmiles(mol)
        except Exception:
            # TODO: Generic exception is ugly
            continue

        # Clean invalid smiles (length, allowed by the dictionary, cannot parse)
        if smiles is None or (
            check_validity and not (sd.allowed(smiles) and len(smiles) < max_len)
        ):
            n_invalid += 1
            continue
        else:
            n_valid += 1

        mol = Chem.MolFromSmiles(smiles)
        properties = pc(mol)

        # Generate numeric encoding of SMILES
        smiles = smiles.strip()
        smiles = sd.encode(smiles)

        numeric_rep = np.zeros(max_len + 2, dtype=np.int32)

        # Add beginning and end tokens
        numeric_rep[0] = sd.char_idx[sd.BEGIN]
        for i in range(len(smiles)):
            numeric_rep[i + 1] = sd.char_idx[smiles[i]]

        numeric_rep[len(smiles) + 1] = sd.char_idx[sd.END]

        dataset.loc[n_valid] = [f.split("_")[1][:-4], smiles] + properties
        fixed_numeric_ds.loc[n_valid] = numeric_rep

        # Compute grammar encoding and grammar encoding mask
        ge, gem = syntax_encode_smiles([smiles])

        grammar_enc.append(ge)
        grammar_enc_masks.append(gem)

    # Remove Duplicates
    if remove_duplicates:
        print("Removing Duplicates")
        duplicates = dataset.duplicated(keep="first")
        duplicate_indices = np.where(duplicates)[0]
        dataset.drop(duplicate_indices, inplace=True)
        fixed_numeric_ds.drop(duplicate_indices, inplace=True)

        print(f"    Removed {len(duplicate_indices)} duplicated rows.")

    print("Saving Data")
    dataset.to_csv(os.path.join(target_path, "QM9_clean.csv"), header=True, index=False)
    fixed_numeric_ds.to_csv(
        os.path.join(target_path, "QM9_fixed_numeric_smiles.csv"),
        header=True,
        index=False,
    )
    # Write grammar encoding and grammar encoding masks
    np.save(
        os.path.join(target_path, "grammar_encodings.npy"),
        np.array(grammar_enc).squeeze(axis=1),
    )
    np.save(
        os.path.join(target_path, "grammar_encoding_masks.npy"),
        np.array(grammar_enc_masks).squeeze(axis=1),
    )

    if verbose:
        print(
            f"   {n_valid} smiles kept and {n_invalid} smiles removed [out of total {n_valid + n_invalid}]"
        )
    return n_valid


def preprocess_zinc(
    prop_names: List[str] = ["LogP"],
    max_len: int = 278,
    target_path: str = "./ZINC250K/",
    source_file_path: str = "./ZINC250K/250k_rndm_zinc_drugs_clean_3.csv",
) -> int:
    """
    Encompasses preprocessing logic for ZINC250k dataset. Generates:
        - Cleaned ZINC canonicalised smiles and properties saved as CSV
        - Cleaned ZINC smiles numericaly encoded smiles as a CSV

    TODO: Add support for grammar encodings

    Source file is csv with the following columns:
        smiles, logP, qed, SAS sourced from kaggle
        [https://www.kaggle.com/datasets/basu369victor/zinc250k, accessed Febuary 2024]

    Args:
        prop_names: List of names of properties to keep.
        max_len: Maximal length of smiles to keep and length to which to pad numeric sequences.
        target_path: Path to existing directory in which to save resulting files.
        source_file_path: Path to source csv file containing SMILES
    Returns:
        int: Number of molecules in the dataset.
    """

    # Load unprocessed dataset csv
    raw_zinc_df = pd.read_csv(source_file_path)

    # Rename kaggle columns to follow our property naming scheme
    raw_zinc_df.rename(
        columns={"logP": "LogP", "qed": "QED", "SAS": "SA"}, inplace=True
    )

    sd = SmilesCharDictionary()
    total_iterations = len(raw_zinc_df["smiles"])
    progress_bar = tqdm(total=total_iterations, desc="Progress", unit="iter")

    # Create dataframes for cleaned SMILES and numeric representations
    clean_ds = pd.DataFrame(
        columns=["ZINC_ID", "SMILES"] + CONFIG.target_props
    )  # + 2 to account for sd.BEGIN and sd.END
    fixed_numeric_ds = pd.DataFrame(
        columns=[("Pos" + str(a)) for a in range(max_len + 2)]
    )

    n_valid = 0

    for ind, row in raw_zinc_df.iterrows():
        progress_bar.update(1)

        # Parse each smile, canonicalize it and fetch relevant properties
        orig_smile = row["smiles"]
        mol = Chem.MolFromSmiles(orig_smile)
        props = list(row[prop_names])
        smiles = Chem.MolToSmiles(mol)

        # Get the numeric representation
        numeric_rep = np.zeros(max_len + 2, dtype=np.int32)

        # Replace 2 character smiles names etc... with one character ones according to encode_dict
        # self.encode_dict = {"Br": 'Y', "Cl": 'X', "Si": 'A', 'Se': 'Z', '@@': 'R', 'se': 'E'}
        # else can't handle two character strings like Br and Cl
        smiles = smiles.strip()
        encoded_smiles = sd.encode(smiles)

        # Attempt to generate numeric encoding
        try:
            numeric_rep[0] = sd.char_idx[sd.BEGIN]
            for i in range(len(encoded_smiles)):
                numeric_rep[i + 1] = sd.char_idx[encoded_smiles[i]]

            numeric_rep[len(encoded_smiles) + 1] = sd.char_idx[sd.END]
        except Exception:
            # TODO: Fix catching generic exception
            print(
                f"Failure to encode: ind {i} | smiles: {smiles} | ecoded_smiles: {encoded_smiles}"
            )
            continue

        clean_ds.loc[n_valid] = [ind, smiles] + props
        fixed_numeric_ds.loc[n_valid] = numeric_rep
        n_valid += 1

    progress_bar.close()

    # Save cleaned dataset csv files
    clean_ds.to_csv(
        os.path.join(target_path, "ZINC_clean.csv"), header=True, index=False
    )
    fixed_numeric_ds.to_csv(
        os.path.join(target_path, "ZINC_fixed_numeric_smiles.csv"),
        header=True,
        index=False,
    )

    # Return number of valid SMILES
    return len(clean_ds["SMILES"])
