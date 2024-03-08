import torch
import torch.nn as nn

# import sys
# sys.path.append('../utils')
# Makes more sense for the training function to instantiate this
# from sd_loss_computation import PerpCalculator

class ConditionalSDLSTM(nn.Module):
    def __init__(self, input_size, property_size, property_names, hidden_size, output_size, n_layers, rnn_dropout, max_rules, rules_dict_size) -> None:
        """
        LSTM model for training on masked grammar encodings as in SD-VAE. Works like the regular CLSTM model
        but holds and records SDLSTM related parameters

        Args:
            max_rules: Maximum no rules to appy
            rules_dict_size: Length of one hots for encodings, input size
            property_size: number of molecule properties
            hidden_size: number of hidden units
            output_size: number of output symbols
            n_layers: number of hidden layers
            rnn_dropout: recurrent dropout
        """

        super().__init__()
        self.input_size = input_size
        self.max_rules = max_rules
        self.rules_dict_size = rules_dict_size
        self.property_size = property_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn_dropout = rnn_dropout
        self.property_names = property_names

        # Model originally expected input to be tensors of SMILES tokens (ints) and embedded them before passing to hiden
        # Since we are working with one hot vectors, we can replace this with a linear layer of the approp dimensionality 

        # LOUIS: TODO: OLD
        # self.encoder = nn.Embedding(input_size, hidden_size)
        self.linear = nn.Linear(input_size, hidden_size)

        self.decoder = nn.Linear(hidden_size, output_size)
        self.rnn = nn.LSTM(input_size=(hidden_size + property_size), hidden_size=(hidden_size), batch_first=True, num_layers=n_layers, dropout=rnn_dropout)
        
        self.init_weights()



    def init_weights(self):
        # encoder / decoder
        # LOUIS: TODO: OLD
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.xavier_uniform_(self.encoder.weight)

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

    def forward(self, x, properties, hidden):
        # LOUIS: TODO: REMOVE
        # embeds = self.encoder(x)
        embeds = self.linear(x)
        output, hidden = self.rnn(torch.cat((embeds, properties.unsqueeze(1).expand(-1, x.shape[1], -1)), -1), hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, bsz, device):
        # LSTM has two hidden states...
        return (torch.zeros(self.n_layers, bsz, self.hidden_size).to(device),
                torch.zeros(self.n_layers, bsz, self.hidden_size).to(device))

    @property
    def config(self):
        return dict(
                    input_size = self.input_size,
                    property_size=self.property_size,
                    property_names= self.property_names,
                    hidden_size=self.hidden_size,
                    output_size=self.output_size,
                    n_layers=self.n_layers,
                    rnn_dropout=self.rnn_dropout,
                    max_rules=self.max_rules,
                    rules_dict_size=self.rules_dict_size)

