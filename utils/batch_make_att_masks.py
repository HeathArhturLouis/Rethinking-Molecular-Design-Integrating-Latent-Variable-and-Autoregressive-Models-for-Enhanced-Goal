#!/usr/bin/env python
import numpy as np
from attribute_tree_decoder import create_tree_decoder
from tree_walker import OnehotBuilder
from mol_util import atom_valence, bond_types, bond_valence, prod, DECISION_DIM, rule_ranges


from config import CONFIG



import cfg_parser
from config import CONFIG
from mol_tree import AnnotatedTree2MolTree
from tree_walker import OnehotBuilder, ConditionalDecoder
from attribute_tree_decoder import create_tree_decoder


def syntax_encode_smiles(smiles_list):

    grammar = cfg_parser.Grammar('../grammar/mol_zinc.grammar')

    cfg_tree_list = []
    for smiles in smiles_list:

        ts = cfg_parser.parse(smiles, grammar)
        # TS[0] contains annotated tree corresponding to SMILES
        assert isinstance(ts, list) and len(ts) == 1

        n = AnnotatedTree2MolTree(ts[0])
        # n is mol_tree.Node

        cfg_tree_list.append(n)

    walker = OnehotBuilder()
    tree_decoder = create_tree_decoder()
    # cfg_tree_list 
    onehot, masks = batch_make_att_masks(cfg_tree_list, tree_decoder, walker, dtype=np.byte)

    return onehot, masks

    

def batch_make_att_masks(node_list, tree_decoder = None, walker = None, dtype=np.byte):
    '''
    node_list: list of mol_tree.Node's for each smile
    tree_decoder: freshly created tree decoder object
    walker = freshly created one hot builder
    '''
    if walker is None:
        walker = OnehotBuilder()
    if tree_decoder is None:
        tree_decoder = create_tree_decoder()

    true_binary = np.zeros((len(node_list), CONFIG.max_decode_steps, DECISION_DIM), dtype=dtype)
    rule_masks = np.zeros((len(node_list), CONFIG.max_decode_steps, DECISION_DIM), dtype=dtype)

    for i in range(len(node_list)):
        node = node_list[i]
        tree_decoder.decode(node, walker)

        true_binary[i, np.arange(walker.num_steps), walker.global_rule_used[:walker.num_steps]] = 1
        true_binary[i, np.arange(walker.num_steps, CONFIG.max_decode_steps), -1] = 1

        for j in range(walker.num_steps):
            rule_masks[i, j, walker.mask_list[j]] = 1

        rule_masks[i, np.arange(walker.num_steps, CONFIG.max_decode_steps), -1] = 1.0

    return true_binary, rule_masks


if __name__ == "__main__":
    #grammar = cfg_parser.Grammar('../grammar/mol_zinc.grammar')
    #ts = cfg_parser.parse('CCO', grammar)
    # TS[0] is 
    #print(type(ts[0]))

    rules, true_masks = syntax_encode_smiles(['CCO'])
    con_decoder = ConditionalDecoder(rules, use_random=False)
