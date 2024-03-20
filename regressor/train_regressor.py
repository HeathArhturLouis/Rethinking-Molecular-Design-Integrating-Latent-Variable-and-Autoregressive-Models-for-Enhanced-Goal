import h5py
import pdb
import numpy as np
import argparse
import random
import sys, os
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


from cmd_args import cmd_args

sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
import cfg_parser as parser
seed = 19260817

## define the model
sys.path.append('%s/../' % os.path.dirname(os.path.realpath('__file__')))
from model import MolVAE, Regressor, train_regressor
sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
from mol_util import rule_ranges, terminal_idxes, DECISION_DIM
from train_util import PerpCalculator
from train_util import raw_logit_to_smile_labels, run_job, Prepare_data
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if not os.path.exists(cmd_args.regressor_saved_dir):
    os.makedirs(cmd_args.regressor_saved_dir)

sys.path.append('%s/../util/' % os.path.dirname(os.path.realpath('__file__')))
getting_data = Prepare_data(cmd_args)

train_binary_x, train_masks, valid_binary_x, valid_masks, train_y, valid_y = getting_data.load_data()
print('num_train: %d\tnum_valid: %d' % (train_y.shape[0], valid_binary_x.shape[0]))

ae = MolVAE()
if cmd_args.mode == 'gpu':
    ae = ae.cuda()
regressor = Regressor(cmd_args.max_decode_steps, cmd_args.latent_dim)
regressor.cuda()

optimizer_regressor = optim.Adam(regressor.parameters(), lr=cmd_args.learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer_regressor, 'min', factor=0.5, patience=5, verbose=True, min_lr=0.001)

# load the pretrained vae model
vae_pretrained_model = cmd_args.vanilla_vae_save_dir + '/epoch-best.model'
if vae_pretrained_model  is not None and vae_pretrained_model != '':
        if os.path.isfile(vae_pretrained_model):
            print('loading model from %s' % vae_pretrained_model)
            ae.load_state_dict(torch.load(vae_pretrained_model))
            
            

sample_idxes = list(range(train_binary_x.shape[0]))
best_valid_loss = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(sample_idxes)

    ## update the reggressor:
    regressor, regressor_loss, reg_1= train_regressor('train', epoch, optimizer_regressor, ae, regressor, sample_idxes,  train_binary_x, train_y)
    print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: average loss %.5f, minibatch %.5f' % (epoch, regressor_loss, reg_1))

    if epoch % 1 == 0:
        _, valid_loss, _= train_regressor('valid', epoch,  None, ae,  regressor, list(range(valid_binary_x.shape[0])), valid_binary_x, valid_y)
        print('>>>>average regressor \033[92mtraining\033[0m of epoch %d: average loss %.5f, minibatch %.5f' % (epoch, regressor_loss, reg_1))
        lr_scheduler.step(valid_loss)
        torch.save(regressor.state_dict(), cmd_args.regressor_saved_dir + '/epoch-%d.model' % epoch)

        if best_valid_loss is None or valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print('saving to best model since this is the best valid loss so far.----')
            torch.save(regressor.state_dict(), cmd_args.regressor_saved_dir + '/epoch-best.model')


