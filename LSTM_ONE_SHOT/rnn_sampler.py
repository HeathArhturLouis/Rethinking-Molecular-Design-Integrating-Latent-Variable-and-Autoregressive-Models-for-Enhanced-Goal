import torch
import numpy as np

from action_sampler import ActionSampler
from rnn_model import ConditionalSmilesRnn
from smiles_char_dict import SmilesCharDictionary


class ConditionalSmilesRnnSampler:
    """
    Samples molecules from an RNN smiles language model
    """

    def __init__(self, device: str, batch_size=64) -> None:
        """
        Args:
            device: cpu | cuda
            batch_size: number of concurrent samples to generate
        """
        self.device = device
        self.batch_size = batch_size
        self.sd = SmilesCharDictionary()

    def sample(self, model: ConditionalSmilesRnn, properties, num_to_sample: int, max_seq_len=100):
        """
        Args:
            model: RNN to sample from
            properties: values of properties to condition on
            num_to_sample: number of samples to produce, for each properties
            max_seq_len: maximum length of the samples
            batch_size: number of concurrent samples to generate

        Returns: an array of SMILES strings, grouped by target properties, with no 
                 beginning nor end symbols
        """

        sampler = ActionSampler(max_batch_size=self.batch_size, max_seq_length=max_seq_len, device=self.device)
        smiles = []

        model.eval()
        with torch.no_grad():
            indices = sampler.sample(model, properties)

            smiles += self.sd.matrix_to_smiles(indices) # np.array(self.sd.matrix_to_smiles(indices.reshape(-1, max_seq_len)))
            return smiles  #.reshape(-1, num_to_sample)


if __name__ == '__main__':
    lstm_params = '../models/NEW_LONG_RUNS/ZINC120/LSTM/model_final_73.302.pt'
    lstm_definit = '../models/NEW_LONG_RUNS/ZINC120/LSTM/model_final_73.302.json'

    from rnn_utils import load_rnn_model


    # lstm_sampler = FastSampler(device = 'cpu', batch_size=64)
    lstm_sampler = ConditionalSmilesRnnSampler(device='cpu', batch_size = 64)
    lstm_model = load_rnn_model(
            model_definition= lstm_definit,
            model_weights = lstm_params,
            device = 'cpu',
            )

    a = lstm_sampler.sample(
        model=lstm_model, 
        properties= torch.tensor([[0.0] for _ in  range(65)]), 
        num_to_sample=65,
        max_seq_len=120)

    print(np.array(a).shape)
    print(a)
