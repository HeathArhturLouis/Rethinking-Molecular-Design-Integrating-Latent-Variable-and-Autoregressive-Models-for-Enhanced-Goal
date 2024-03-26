import torch
import torch.nn as nn


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

    def forward(self, x, properties, hidden):
        '''
        Without Teacher Forcing
        '''
        device = properties.device
        
        max_length = x.shape[1]
        # Shape: Batch size
        initial_input = torch.tensor([[0] for _ in range(properties.shape[0])], device=device)

        batch_size = properties.size(0)

        hidden = self.init_hidden(batch_size, device)

        # Prepare the initial input token, which could be a 'start' token
        inputs = self.encoder(initial_input)  # Assuming initial_input is already a tensor of token IDs

        outputs = []

        for _ in range(max_length):
            # Combine the inputs with the properties
            combined_inputs = torch.cat((inputs, properties.unsqueeze(1).expand(-1, 1, -1)), -1)

            # Pass the combined input through the RNN
            rnn_output, hidden = self.rnn(combined_inputs, hidden)

            # Decode to get the logits for the next token
            logits = self.decoder(rnn_output)

            # Store the logits
            outputs.append(logits)

            # Determine the next input token
            next_input_token_ids = torch.argmax(logits, dim=-1)
            inputs = self.encoder(next_input_token_ids)

            # Implement your stopping criterion here

        # Concatenate all logits from each step to form the output sequence
        outputs = torch.cat(outputs, dim=1)
        
        return outputs, hidden

    '''
    def forward(self, x, properties, hidden):
        # inp is: 20x101 batch_size x max_seq_len
        # properties is: 20x1 batch_size [property logP]
        # input size is 47
        embeds = self.encoder(x)
        output, hidden = self.rnn(torch.cat((embeds, properties.unsqueeze(1).expand(-1,x.shape[1], -1)), -1), hidden)
        output = self.decoder(output)
        return output, hidden

    '''

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
