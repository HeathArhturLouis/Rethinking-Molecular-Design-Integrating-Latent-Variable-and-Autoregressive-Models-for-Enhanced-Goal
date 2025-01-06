"""
CLI configuration interface for ZINC250k preprocessing.
"""

import argparse
from preprocess_dataset import preprocess_zinc, generate_data_splits

if __name__ == "__main__":
    # Define CLI argument parser and arguments
    parser = argparse.ArgumentParser(
        description="CLI configuration interface for ZINC250k preprocessing."
    )

    parser.add_argument(
        "--ZINC_target_path",
        type=str,
        default="./zinc250k_120",
        help="Path to dir with ZINC data to operate in.",
    )
    parser.add_argument(
        "--ZINC_source_path",
        type=str,
        default="./ZINC250K/250k_rndm_zinc_drugs_clean_3.csv",
        help="Path to raw ZINC240k csv with properties and SMILES.",
    )

    parser.add_argument(
        "--ZINC_smile_len",
        type=int,
        default=120,
        help="Length to which to pad ZINC token sequences. SMILES longer than this will be discarded.",
    )
    parser.add_argument(
        "--ZINC_prop_names", type=list, default=["LogP"], help="List of property names."
    )
    parser.add_argument(
        "--ZINC_test_size",
        type=int,
        default=10000,
        help="Number of mols in train set. (remainder will be used for train)",
    )
    parser.add_argument(
        "--ZINC_validation_size",
        type=int,
        default=10000,
        help="Number of mols in validation set. (remainder will be used for train)",
    )
    parser.add_argument(
        "--ZINC_split_seed",
        type=int,
        default=42,
        help="Seed for determining random splits",
    )

    args = parser.parse_args()

    # Call preprocessing function
    print("Cleaning ZINC Data")
    n_valid = preprocess_zinc(
        prop_names=args.ZINC_prop_names,
        max_len=args.ZINC_smile_len,
        target_path=args.ZINC_target_path,
        source_file_path=args.ZINC_source_path,
    )

    print("Generating Data Splits")
    generate_data_splits(
        args.ZINC_target_path,
        n_valid,
        args.ZINC_test_size,
        args.ZINC_validation_size,
        random_state=args.ZINC_split_seed,
    )
