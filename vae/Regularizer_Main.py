from train_vae_with_regularizer import VanillaVAETrainerReg
from model_vanilla_vae import VanillaMolVAE
from rnn_utils import load_smiles_and_properties, load_model

import argparse


if __name__ == "__main__":
        # Initialize the parser
        parser = argparse.ArgumentParser(description="Set up model paths and device")

        # Adding arguments
        parser.add_argument("--data_path", type=str, default='../data/QM9/', help="Path to the data directory")
        parser.add_argument("--regularizer_model_path", type=str, default='../models/LSTM_QM9/batch_size_64_2/LSTM_20_1.190', help="Path to the regularizer model")
        parser.add_argument("--prior_model_path", type=str, default=None, help="Path to the prior model")
        parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'], help="Compute device")
        parser.add_argument("--save_dir", type=str, default='../models/REG_VAE_QM9_3L/', help="Output path to.")
        parser.add_argument("--reg_weight", type=float, default=0.1, help="Weight for the regularizer component of the loss")

        # Parse the arguments
        args = parser.parse_args()

        # Accessing the arguments
        data_path = args.data_path
        regularizer_model_path = args.regularizer_model_path
        prior_model_path = args.prior_model_path
        device = args.device
        reg_weight = args.reg_weight

        if prior_model_path == None:
            prior_model = None
        else:
            prior_model = load_model(model_class=VanillaMolVAE, 
                                    model_definition=prior_model_path + '.json',
                                    model_weights=prior_model_path + '.pt',
                                    device=device)

        # Load Property Data and Masks and create splits
        property_names, train_x_enc, train_y, valid_x_enc, valid_y = load_smiles_and_properties(data_path)

        # Instantiate Trainer Class
        trainer =  VanillaVAETrainerReg(
                regularizer_model_path = regularizer_model_path + '.pt',
                regularizer_param_path = regularizer_model_path + '.json',
                regularizer_weight = reg_weight,
                train_x_enc=train_x_enc,
                train_y=train_y,
                valid_x_enc=valid_x_enc,
                valid_y=valid_y,
                property_names=["LogP"],
                device=device,
                batch_size=64,
                # TODO: LOUIS: This is a bit sus, it's 56 in the other one
                latent_dim=56,
                beta=1.0,
                max_seq_len=101,
                eps_std=0.01,
                model=prior_model,
                decoder_embedding_dim=47,
                learning_rate = 1e-3,
                model_save_dir = '../models/REG_VAE_QM9_3L/', 
                valid_every_n_epochs = 1, 
                save_every_n_val_cycles = 3, 
                max_epochs = 100
                )

        # Save some memory in case we're running on CPU, delete external refs to dead objects
        del train_x_enc, valid_x_enc, train_y, valid_y
        
        trainer.fit()