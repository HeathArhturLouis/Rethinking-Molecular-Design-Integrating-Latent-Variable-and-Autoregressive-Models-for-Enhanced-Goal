from typing import Type

import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Distribution

from rnn_model import ConditionalSmilesRnn
from rnn_utils import rnn_start_token_vector


class ActionSampler:
    """
    Sampler for a SmilesRNN models.

    Does not return SMILES strings directly, but instead the actions (i.e. which SMILES character to select).
    Those values are more general and are for instance necessary for other RL algorithms.

    The class will sample the RNN model multiple times if the number of desired samples is larger than the
    maximal allowed batch size.
    """

    def __init__(self, max_batch_size, max_seq_length, device,
                 distribution_cls: Type[Distribution] = None) -> None:
        """
        Args:
            max_batch_size: maximal batch size for the RNN model
            max_seq_length: max length for a sampled SMILES string
            device: cuda | cpu
            distribution_cls: distribution type to sample from. If None, will be a multinomial distribution. Useful for testing purposes.
        """
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.device = device

        self.distribution_cls = Categorical if distribution_cls is None else distribution_cls

    def sample(self, model: ConditionalSmilesRnn, properties, num_samples: int) -> torch.Tensor:
        """
        Samples a specified number of actions from an RNN model based on a multinomial distribution.

        Args:
            model: Smiles RNN model to sample from
            properties: Values of properties to target, either:
                        a vector of size (num_properties, ), or
                        a matrix of size (num_targets, num_properties)
            num_samples: Number of samples to generate

        Returns:
            tensor of actions (num_samples x num_targets x max_seq_length)
        """

        properties = torch.FloatTensor(properties)
        
        # Just assume properties defined Correctly
        #if len(properties.shape) == 1:
        #    properties.unsqueeze(0)
        #assert len(properties.shape) == 2
        num_properties = properties.shape[-1]
        
        # Resize properties in order to batch across num_samples and num_targets
        expanded_properties = properties.unsqueeze(1).expand(-1, num_samples, num_properties).reshape(-1, num_properties)

        # Round up division to get the number of batches that are necessary:
        total_samples = expanded_properties.shape[0]
        number_batches = (total_samples + self.max_batch_size - 1) // self.max_batch_size
        remaining_samples = total_samples

        actions = torch.LongTensor(total_samples, self.max_seq_length).to(self.device)

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            batch_properties = expanded_properties[batch_start:batch_end].to(self.device)
            actions[batch_start:batch_end, :] = self._sample_batch(model, batch_properties, batch_size)

            batch_start += batch_size
            remaining_samples -= batch_size

        return actions.reshape(-1, num_samples, self.max_seq_length)

    def _sample_batch(self, model: ConditionalSmilesRnn, properties, batch_size: int) -> torch.Tensor:
        """
        Samples a batch of actions based on a multinomial distribution.

        Args:
            model: Smiles RNN model to sample from
            num_samples: Number of samples to generate

        Returns:
            tensor of actions (batch_size x max_seq_length)
        """
        hidden = model.init_hidden(batch_size, self.device)

        inp = rnn_start_token_vector(batch_size, self.device)
        
        
        actions = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(self.device)
#         properties = properties.unsqueeze(0).expand(batch_size, -1)

        actions = model.forward(self, None, properties, hidden, use_teacher_forcing = False, sampling = False, return_actions = True)

        '''
        for char in range(self.max_seq_length):
            output, hidden = model(inp, properties, hidden)

            prob = F.softmax(output, dim=2)
            distribution = self.distribution_cls(probs=prob)
            action = distribution.sample()

            actions[:, char] = action.squeeze()

            inp = action
        '''

        return actions


