import torch
import torch.nn as nn

import torch.nn.functional as F

from torch.distributions import Categorical


import sys

sys.path.append('../utils')
from smiles_char_dict import SmilesCharDictionary


class ConditionalSmilesRnn(nn.Module):
    def __init__(self, input_size, property_size, property_names, hidden_size, output_size, n_layers, rnn_dropout) -> None:
        """
            RNN language model for SMILES, defined conditional on molecule properties

        Args:
            input_size: number of input symbols
            property_size: number of molecule properties
            hidden_size: number of hidden units
            output_size: number of output symbols
            n_layers: number of hidden layers
            rnn_dropout: recurrent dropout
        """
        super().__init__()
        self.input_size = input_size
        self.property_size = property_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.property_names = property_names


        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(hidden_size + property_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=rnn_dropout)
        self.init_weights()

        self.distribution_cls = Categorical

        self.sd = SmilesCharDictionary()

    def init_weights(self):
        # encoder / decoder
        nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

        # RNN
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)):int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, x, properties, hidden, use_teacher_forcing = False, sampling = True, return_actions = False, return_both = False, seq_len = 100):
        '''
        Expects initial token in input when using teacher forcing
        
        Returns both actions and logits without the initial token!

        '''
        # x is: 20x101 batch_size x max_seq_len
        # properties is: 20x1 batch_size [property logP]
        # input size is 47
        '''
        if teacher_forcing:
            embeds = self.encoder(x)
            output, hidden = self.rnn(torch.cat((embeds, properties.unsqueeze(1).expand(-1,x.shape[1], -1)), -1), hidden)
            output = self.decoder(output)

            # Since output is of size 20x101, it's creating the output at 20xX from the inputs at 20x(X-1) ergo it's using teacher forcing
            return output, hidden
        '''
        outputs = []
        actions = []

        # Expects initial tokens

        # input = rnn_start_token_vector(self.batch_size, self.device)
        b_size = properties.shape[0]
        # TODO: FIX HARD CODED VALUE

        input = torch.LongTensor(b_size, 1).fill_(self.sd.begin_idx).to(properties.device)
        
        # input = x[:, 0].unsqueeze(1)  # Start with the first input timestep
            
        for t in range(1, seq_len): # Range 0 -- 101
            raw_out, hidden = self.rnn(torch.cat((self.encoder(input), properties.unsqueeze(1)), -1), hidden)
            logits = self.decoder(raw_out)
            outputs.append(logits)
            
            if use_teacher_forcing:
                action = x[:, t].unsqueeze(1)
                input = action
            else:
                # Sample output from cateogrical distribution constructed from logits, track selected output and sample next action
                if sampling:
                    prob = F.softmax(logits, dim=2)
                    distribution = self.distribution_cls(probs=prob)
                    action = distribution.sample()
                else:
                    action = torch.argmax(logits, dim=-1)

                input = action
            actions.append(action)

        outputs = torch.cat(outputs, dim=1)
        actions = torch.cat(actions, dim=1)

        if return_both:
            return actions, outputs
        elif return_actions:
            return actions, hidden    
        else:
            return outputs, hidden

    

    def init_hidden(self, bsz, device):
        # LSTM has two hidden states...
        return (torch.zeros(self.n_layers, bsz, self.hidden_size).to(device),
                torch.zeros(self.n_layers, bsz, self.hidden_size).to(device))

    @property
    def config(self):
        return dict(input_size=self.input_size,
                    property_size=self.property_size,
                    property_names= self.property_names,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    n_layers=self.n_layers,
                    rnn_dropout=self.rnn_dropout)
