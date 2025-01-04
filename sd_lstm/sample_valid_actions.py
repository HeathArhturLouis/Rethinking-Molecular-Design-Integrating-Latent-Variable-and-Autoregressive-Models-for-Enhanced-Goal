import sys

import numpy as np

sys.path.append('../utils')
from mol_tree import AnnotatedTree2MolTree, get_smiles_from_tree, Node
from tree_walker import ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder
from attribute_tree_generator import create_sequential_tree_decoder, FETCH_TOKEN
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
            tree_decoder = create_tree_decoder() # is of type AttMolGraphDecoder
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


def step_through_logits_test(partial_logit_sequence):
    '''
    DELETE ME: Test weather we can property edit walker from the outside
    '''
    if not len(partial_logit_sequence) > 0:
        walker = ConditionalDecoder(np.array([partial_logit_sequence]), use_random=False)
    else:
        walker = ConditionalDecoder(np.array([partial_logit_sequence[0]]), use_random=False)
    
    new_t = Node('smiles')

    tree_decoder = create_sequential_tree_decoder()
    tree_decoder_generator = tree_decoder.decode(new_t, walker)

    # Step through current rules
    for logits in partial_logit_sequence[1:]:
        next_op = next(tree_decoder_generator)
        if next_op == FETCH_TOKEN:
            print('Next Mask')
            print(walker.mask_list[-1])

            # Append next logits
            walker.raw_logits = np.append( walker.raw_logits, [logits], axis=0)
        else: # Construction has terminated, no need to update tree anymore, can just get smiles out of it 
            # will ignore rest of logits if provided
            print('Sequence already terminated')
            return [1] * 80
    print('Seems to have gone fine')





def baseline(raw_logits, use_random=False):
    walker = ConditionalDecoder(np.squeeze(raw_logits), use_random=use_random) 
    new_t = Node('smiles')
    tree_decoder = create_tree_decoder()
    tree_decoder.decode(new_t, walker)
    print('Seems to have gone well')


def next_mask_from_logits(partial_logit_sequence, use_random=False):
    '''
    partial_logit_sequence contains part but not all of the logits for generation, returns the

    WARNING: use_random NEEDS to be False here since otherwise sequence might diverge and mask might
    not correspond to selected inputs

    ASSUME: partial_logit_sequence[0] is start token logit [1, 0, 0, 0, 0 \ldots]
    '''
    
    walker = ConditionalDecoder(partial_logit_sequence, use_random=use_random)
    new_t = Node('smiles')
    tree_decoder = create_sequential_tree_decoder()
    tree_decoder_generator = tree_decoder.decode(new_t, walker)


    # Step through current rules
    for logit in partial_logit_sequence:
        # 1. Get next mask
        next_op = next(tree_decoder_generator)
        if next_op[1] == FETCH_TOKEN:
            # print('Passing Token')
            pass
        else:
            # print('Sequence already terminated')
            return 'ALL'

    '''FIRST ITERATION:
    1. Next to get mask for first logit 
    2. Update walker with first token
    PROCESS:
    1. Next to get next mask
    2. Compute logits, mask them and add them to the walker
    3. Continue execution
    '''
    CHECK_YIELD = next(tree_decoder_generator)

    if CHECK_YIELD[1] == FETCH_TOKEN:
        # Pretend We're sampling model
        # Pretend we're sampling tokens
        # Pretend we're updating walker
        return CHECK_YIELD[0]
    else:
        return 'ALL'


if __name__ == '__main__':
    # 
    logits = [0, 69, 2, 5, 57, 67,  2,  5, 57, 52, 79] + ([79] * (100 - 11))

    ddim = 80

    # logits = [0, 68, 2, 7, 57, 68, 2, 5, 59, 78, 55, 61]
    logits = [0, 68, 2, 6, 57, 69, 2, 5, 57, 68, 1, 18, 20, 29, 6, 24, 26, 48, 44]
    next_log = [-1]

    logits = np.eye(ddim)[logits]
    # print(logits.shape[1])
    next_log = np.eye(ddim)[next_log][0]
    next_ind = np.argmax(next_log)

    next_mask = next_mask_from_logits(logits)
    print(f'{next_ind} should be in {next_mask}')


    # For [0, 69, 2, 5] gives [ 4  5  6  7  8  9 10 11 12 13]
    # For [0, 69, 2] gives (5)
    # For [0, 69, 2, 5, 57, 67] gives [1,2,3] (2)
    # It's giving the current last mask, not the next
       
    # [print(raw_logit_to_next_iter_masks(logits)) for _ in range(100)]
    # baseline(logits)