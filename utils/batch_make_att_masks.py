#!/usr/bin/env python
import numpy as np
from attribute_tree_decoder import create_tree_decoder
from tree_walker import OnehotBuilder
from mol_util import atom_valence, bond_types, bond_valence, prod, DECISION_DIM, rule_ranges


from config import CONFIG

def batch_make_att_masks(node_list, tree_decoder = None, walker = None, dtype=np.byte):
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
