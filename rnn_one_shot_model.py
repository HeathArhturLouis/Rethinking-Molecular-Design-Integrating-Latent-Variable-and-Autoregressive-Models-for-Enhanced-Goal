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
        
        # Input should just be property_size --> no previous input!
        # self.rnn = nn.LSTM(hidden_size + property_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=rnn_dropout)
        self.rnn = nn.LSTM(property_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=rnn_dropout)

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

        Will accept inputs but will ignore them since ONE-Shot to maintain interface with rest of code
        '''

        outputs = []
        actions = []

        b_size = properties.shape[0]

        # Extent properties
        properties = properties.unsqueeze(1).repeat(1, seq_len - 1, 1)

        rnn_output, hidden = self.rnn(properties, hidden)
        
        # Generate logits
        output = self.decoder(rnn_output) 

        # Output is logits is 64x37x47 and contains logits

        if (not return_actions) and (not return_both):
            return output, hidden # return just logits

        probabilities = F.softmax(output, dim=-1)
        # Sample actions
        if sampling:
            actions =  self.distribution_cls(probs=probabilities).sample()
        else:
            actions = torch.argmax(probabilities, dim=-1)

        if return_both:
            return actions, outputs
        else: # return actions
            return actions, hidden


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
