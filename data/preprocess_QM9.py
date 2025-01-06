"""
CLI configuration interface for QM9 preprocessing.
"""

import argparse
from preprocess_dataset import preprocess_qm9, generate_data_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI configuration interface for ZINC250k preprocessing.")

    parser.add_argument(
        "--QM9_raw_data_path",
        type=str,
        default="./QM9/xyzfiles",
        help="Path to dir containing raw QM9 data.",
    )
    parser.add_argument(
        "--QM9_target_path",
        type=str,
        default="./qm9_36/",
        help="Path to dir to store QM9 csv and split indecies.",
    )
    parser.add_argument(
        "--QM9_max_smile_len",
        type=int,
        default=36,
        help="Length to pad to. SMILES longer than this will be discarded.",
    )
    parser.add_argument(
        "--QM9_prop_names",
        type=list,
        default=['LogP'],
        help="List of properties to overrride the one found in config",
    )
    parser.add_argument(
        "--QM9_test_size",
        type=int,
        default=10000,
        help="Number of mols in train set. (remainder will be used for train)",
    )
    parser.add_argument(
        "--QM9_validation_size",
        type=int,
        default=10000,
        help="Number of mols in validation set. (remainder will be used for train)",
    )
    parser.add_argument(
        "--QM9_split_seed",
        type=int,
        default=42,
        help="Seed for determining random splits",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Path to dir to store QM9 csv and split indecies.",
    )

    args = parser.parse_args()

    print("Cleaning QM9 Data and Generating Properties.")
    n_valid = preprocess_qm9(
        raw_data_dir=args.QM9_raw_data_path,
        target_path=args.QM9_target_path,
        verbose=args.verbose,
        prop_names=args.QM9_prop_names,
        max_len=36,
    )

    print("Generating Test-Train-Validation Splits.")
    generate_data_splits(
        target_path=args.QM9_target_path,
        n_molecules=n_valid,
        test_size=args.QM9_test_size,
        validation_size=args.QM9_validation_size,
        random_state=args.QM9_split_seed,
    )
