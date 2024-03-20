import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for molecule vae')
cmd_opt.add_argument('-vanilla_vae_save_dir', default='../supervised_vae/vanilla_supervised_vae', help='pretrained vanilla supervised vae dir')
cmd_opt.add_argument('-training_data_dir', default='../../data/data_100', help='training data directory')
cmd_opt.add_argument('-info_folder', default = '../../../dropbox', help = 'the location of context free grammar')
cmd_opt.add_argument('-regressor_saved_dir', default='./regressor_pretrained', help='pretrained regressor dir')

cmd_opt.add_argument('-batch_size', type=int, default=300, help='minibatch size')
cmd_opt.add_argument('-num_epochs', type=int, default= 300, help='number of epochs')
cmd_opt.add_argument('-output_dim', default=1)
cmd_opt.add_argument('-kl_coeff', type=float, default=1, help='coefficient for kl divergence used in vae')

cmd_opt.add_argument('-mode', default='gpu', help='cpu/gpu')
cmd_opt.add_argument('-save_dir', default='./vanilla_supervised_vae', help='result output root')
cmd_opt.add_argument('-saved_model', default=None, help='start from existing model')
cmd_opt.add_argument('-grammar_file', default='../../util/mol_zinc.grammar')

cmd_opt.add_argument('-data_dump', help='location of h5 file')
cmd_opt.add_argument('-encoder_type', default='cnn', help='choose encoder from [tree_lstm | s2v]')
cmd_opt.add_argument('-ae_type', default='vae', help='choose ae arch from [autoenc | vae]')
cmd_opt.add_argument('-rnn_type', default='gru', help='choose rnn cell from [gru | sru]')
cmd_opt.add_argument('-loss_type', default='perplexity', help='choose loss from [perplexity | binary]')
cmd_opt.add_argument('-max_decode_steps', type=int, default=100, help='maximum steps for making decoding decisions')
cmd_opt.add_argument('-seed', type=int, default=1, help='random seed')
cmd_opt.add_argument('-skip_deter', type=int, default=0, help='skip deterministic position')
cmd_opt.add_argument('-bondcompact', type=int, default=0, help='compact ringbond representation or not')
cmd_opt.add_argument('-latent_dim', type=int, default=56, help='minibatch size')
cmd_opt.add_argument('-data_gen_threads', type=int, help='number of threads for data generation')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-prob_fix', type=float, default=0, help='numerical problem')
cmd_opt.add_argument('-eps_std', type=float, default=0.01, help='the standard deviation used in reparameterization tric')

##data processing params
cmd_opt.add_argument('-data_save_dir', default='', help = 'directory to save the cleaned data')
cmd_opt.add_argument('-smiles_file')

cmd_args, _ = cmd_opt.parse_known_args()
