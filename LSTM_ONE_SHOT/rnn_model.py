import torch
import torch.nn as nn

import sys
sys.path.append('../utils')

from smiles_char_dict import SmilesCharDictionary



sd = SmilesCharDictionary()
char_num = sd.get_char_num()


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

        # self.encoder = nn.Embedding(input_size, hidden_size)
        # Modified to accept one hot encodings
        self.encoder = nn.Linear(output_size, hidden_size)

        self.decoder = nn.Linear(hidden_size, output_size)

        self.rnn = nn.LSTM(hidden_size + property_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=rnn_dropout)
        self.init_weights()

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

    def forward(self, x, properties):
        '''
        Without Teacher Forcing

        Ignore x, since we're not teacher forcing
        -  Initialize one hots with zero in them
        -  Feed one hots through recursively to obtain logits
        '''

        device = properties.device

        max_length = x.shape[1]
        # Hard coded for speed
        initial_input = torch.tensor([([0] * char_num) for _ in range(properties.shape[0])], device=device, dtype=torch.float)
        initial_input[:, 1] = 1

        batch_size = properties.size(0)

        hidden = self.init_hidden(batch_size, device)

        # Initial input is batch_size x decision_dim

        # Prepare the initial input token, which could be a 'start' token
        inputs = self.encoder(initial_input) # Assuming initial_input is already a tensor of token IDs

        outputs = torch.zeros([batch_size, max_length, char_num]).to(device)

        # batch_size x msl x dec_dim

        # outputs[:, 0, :] = initial_input.clone()

        # batch_size x seq_len x decision dim

        # Don't return start token
        for step in range(0, max_length):

            # inputs is 64, 512
            # properties is 64, 1

            combined_inputs = torch.cat([inputs, properties], 1)

            # Pass the combined input through the RNN
            rnn_output, hidden = self.rnn(combined_inputs, hidden)

            # Decode to get the logits for the next token
            logits = self.decoder(rnn_output)

            # Store the logits
            outputs[:, step, :] = logits

            # Determine the next input token

            # Feed raw logits directly back into encoder
            inputs = self.encoder(logits)
            # inputs = logits

            # Implement your stopping criterion here

        return outputs



    def init_hidden(self, bsz, device):
        # LSTM has two hidden states...
        return (torch.zeros(self.n_layers, self.hidden_size).to(device),
                torch.zeros(self.n_layers, self.hidden_size).to(device))

    @property
    def config(self):
        return dict(input_size=self.input_size,
                    property_size=self.property_size,
                    property_names= self.property_names,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    n_layers=self.n_layers,
                    rnn_dropout=self.rnn_dropout)
