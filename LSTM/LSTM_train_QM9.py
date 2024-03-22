import pickle
import gzip
import argparse
import os
import sys

from smiles_rnn_distribution_learner import SmilesRnnDistributionLearner

if __name__ == '__main__':
    # setup_default_logger()
    parser = argparse.ArgumentParser(description='Distribution learning benchmark for SMILES LSTM',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='../data/QM9/') # Expects data in README format (QM9_clean.csv, data_splits.npy)

    '''
    parser.add_argument('--train_data', default='./data/QM9_clean_smi_train_smile.npz', #Formerly npz
                        help='Full path to SMILES file containing training data')
    parser.add_argument('--valid_data', default='',
                        help='Full path to SMILES file containing validation data')
    '''

    parser.add_argument('--batch_size', default=64, type=int, help='Size of a mini-batch for gradient descent')
    parser.add_argument('--valid_every', default=1000, type=int, help='Validate every so many batches')
    parser.add_argument('--print_every', default=100, type=int, help='Report every so many batches')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of training epochs') ## Originally this was 100
    parser.add_argument('--max_len', default=100, type=int, help='Max length of a SMILES string')
    parser.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer')
    parser.add_argument('--n_layers', default=3, type=int, help='Number of layers for training')
    parser.add_argument('--rnn_dropout', default=0.2, type=float, help='Dropout value for LSTM')
    parser.add_argument('--lr', default=1e-3, type=float, help='LSTM learning rate')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    #parser.add_argument('--prop_model', default="../../data/QM9/prior.pkl.gz", help='Saved model for properties distribution')

    parser.add_argument('--output_dir', default='../models/LSTM_QM9/', help='Output directory')

    args = parser.parse_args()

    args.output_dir = args.output_dir + '/batch_size_' + str(args.batch_size) + '_2'
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # In SupervisedVAE the validation data is the first 10000 data samples from train file
    
    trainer = SmilesRnnDistributionLearner(data_set = "QM9",
                                           #graph_logger= graph_logger,
                                           output_dir=args.output_dir,
                                           n_epochs=args.n_epochs,
                                           hidden_size=args.hidden_size,
                                           n_layers=args.n_layers,
                                           max_len=args.max_len,
                                           batch_size=args.batch_size,
                                           rnn_dropout=args.rnn_dropout,
                                           lr=args.lr,
                                           valid_every=args.valid_every)

#     training_set_file = args.train_data
#     validation_set_file = args.valid_data
# 
#     with open(training_set_file) as f:
#         train_list = f.readlines()
# 
#     with open(validation_set_file) as f:
#         valid_list = f.readlines()

    trainer.train(args.data_path)  #LOUIS, training_set=args.train_data, validation_set = args.valid_data)

    print(f'All done, your trained model is in {args.output_dir}')                                                                                                                                                                                 
