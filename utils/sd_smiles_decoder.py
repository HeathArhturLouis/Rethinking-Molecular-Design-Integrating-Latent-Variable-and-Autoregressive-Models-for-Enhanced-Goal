import sys

import numpy as np

sys.path.append('../utils')
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from tree_walker import ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder
import sascorer

def raw_logit_to_smiles(raw_logits, use_random = False, quiet=False):
    '''
    Input format:
        raw_logits - [timesteps, batchsize, prediction_dimension]

    Take raw logits as input and convert into a SMILES string
    Equivalently can also take one hot encodings as inputs and interpret them as logits to decode strings
    '''

    # TODO: convert from tensor if expecting tensor
    # Torch tensor is a for the tree decoder method down the line
    if use_random == True:
        assert isinstance(raw_logits, np.ndarray)

    # TODO: Major clean up
    index = []
    generations = []

    for i in range(raw_logits.shape[1]):
        pred_logits = raw_logits[:, i, :]
        walker = ConditionalDecoder(np.squeeze(pred_logits), use_random)
        new_t = Node('smiles')
        try:
            tree_decoder = create_tree_decoder()
            tree_decoder.decode(new_t, walker)
            sampled = get_smiles_from_tree(new_t)
            
        except Exception as ex:
            if not type(ex).__name__ == 'DecodingLimitExceeded' and not quiet:
                print('Warning, decoder failed with', ex)

            # failed. output None
            sampled = None
 
        if sampled is None:
            continue
        
        # we generated a good one
        generations.append(sampled)


    return generations


