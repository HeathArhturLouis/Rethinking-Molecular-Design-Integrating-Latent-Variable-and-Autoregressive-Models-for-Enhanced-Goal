#!/usr/bin/env python
from __future__ import print_function

import os
import sys
import csv
import numpy as np
import math
import random
import torch 
from collections import defaultdict

from mol_util import avail_atoms, atom_valence, bond_types, bond_valence, prod, MAX_NESTED_BONDS, rule_ranges, TOTAL_NUM_RULES
from mol_tree import Node, get_smiles_from_tree, AnnotatedTree2MolTree


#sys.path.append( '%s/../' % os.path.dirname(os.path.realpath(__file__)) )
sys.path.append('../')
from config import CONFIG

# Indicator, when yielded means a rule should be fetched from the walker
FETCH_TOKEN = 'fetch me'
RETURN_TOKEN = 'return me'

class RingBond(object):
    def __init__(self, pos, b_type):
        self.pos = pos
        self.b_type = b_type

class AttMolGraphDecoderGenerator(object):
    def __init__(self):
        self.reset_state()
        self.final_node = None
        # self.sequential = sequential

    def reset_state(self):
        self.atom_num = 0
        self.matched_bonds = set()
        self.open_rings = {}
        self.sameatom_bonds = defaultdict(set)

    def get_node(self, node, new_sym, pos):
        if node.is_created():
            assert pos < len(node.children)
            ans = node.children[pos]
            ans.init_atts()
            if ans.symbol != new_sym:
                print(ans.symbol, new_sym)
            assert ans.symbol == new_sym
            return ans
        return Node(new_sym, node)

    def get_mask_rule(self, node, sub_ranges=None):
        '''
        Returns the mask indexes at next step without sampling from walker
        '''
        g_range = rule_ranges[node.symbol]
        idxes = np.arange(g_range[0], g_range[1])
        if sub_ranges is not None:
            idxes = idxes[sub_ranges]
        return idxes

    def rand_rule(self, node, sub_ranges = None):
        # Determines range of rules that can be applied based on the node type
        g_range = rule_ranges[node.symbol]
        idxes = np.arange(g_range[0], g_range[1])
        if sub_ranges is not None:
            idxes = idxes[sub_ranges]

        # YIELD 1: Here I can return idxex!
        assert len(idxes)
        if len(idxes) == 1 and CONFIG.skip_deter:
            result = 0
        else:
            result = self.walker.sample_index_with_mask(node, idxes)

        if sub_ranges is not None:
            new_idx = sub_ranges[result]
        else:
            new_idx = result
        
        if node.rule_used is not None:
            assert node.rule_used == new_idx
        else:
            node.rule_used = new_idx
        # YIELD 2: Here I return the value!
        return node.rule_used

    def rand_att(self, node, candidates):
        # Here I can yield candidates as the next step mask
        if len(candidates) == 1 and CONFIG.skip_deter:
            att_idx = candidates[0]
        else:
            att_idx = self.walker.sample_att(node, candidates)
        if not hasattr(node, 'bond_idx'):
            node.bond_idx = att_idx
        else:
            assert node.bond_idx == att_idx

        return att_idx

    def ring_valid(self, r, pre_pos, remain):
        p = (self.open_rings[r].pos, self.atom_num - 1)
        if self.open_rings[r].pos == self.atom_num - 1:
            return False
        if self.open_rings[r].pos == pre_pos:
            return False
        if p in self.matched_bonds:
            return False
        if bond_valence[self.open_rings[r].b_type] > remain:
            return False
        return True

    def maximum_match(self, pre_pos, remain):
        if remain == 0:
            return 0
        cur_pos = self.atom_num - 1
        s = set()
        ans = 0
        rest = remain
        for cost in range(1, 4):
            for r in self.open_rings:
                if bond_valence[self.open_rings[r].b_type] != cost:
                    continue
                if self.ring_valid(r, pre_pos, rest) and not self.open_rings[r].pos in s:
                    s.add(self.open_rings[r].pos)
                    rest -= 1
                    ans += 1
                    assert rest >= 0
                    if rest == 0:
                        return ans
        return ans
            
    def tree_generator(self, node, left_conn = False, right_conn = False, cap_remain = None, ref_symbol = None, is_last = None):
        '''
        Generates a molecular tree

        cap_remaining <-- capacity for some operation

        Louis:
        Increase n_sequential with each subsequent decoding step
        When it equals max_steps, break out of the function and return the current node

        Either yields or 'returns' value, if recieves yield, should pass it back up
        if recieves return should continue execution considering this value
        '''

        assert is_last is not None
        if node.symbol in ['bond', 'BB', 'branch', 'BAC', 'BAH', 'charge', 'hcount']:
            assert cap_remain is not None
        
        if node.symbol == 'chain':

            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)
            
            
            a = self.get_node(node, 'branched_atom', 0)
            node.add_child(a)
            # self.tree_generator(a, left_conn, right_conn, is_last = is_last)
            if rule == 0: # chain -> branched_atom
                for value in self.tree_generator(a, left_conn, right_conn, is_last = is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value

                node.left_remain = a.left_remain
                node.right_remain = a.right_remain
                node.single_atom = True
            else:


                # res = self.tree_generator(a, left_conn, True, is_last = False)
                for value in self.tree_generator(a, left_conn, True, is_last = False):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        res = value[0]
                    else:
                        yield value
        

                c = self.get_node(node, 'chain', -1)
                c.pre_node = a.atom_pos
                assert c.pre_node is not None


                # self.tree_generator(c, True, right_conn, is_last = is_last)
                for value in self.tree_generator(c, True, right_conn, is_last = is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value
                

                cost = 0
                if rule == 2: # chain -> branched_atom bond chain
                    b = self.get_node(node, 'bond', 1)


                    # self.tree_generator(b, cap_remain = min(c.left_remain, a.right_remain) + 1, is_last=is_last)
                    for value in self.tree_generator(b, cap_remain = min(c.left_remain, a.right_remain) + 1, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            continue
                        else:
                            yield value

                    
                    cost = bond_valence[b.children[0].symbol] - 1
                    node.add_child(b)
                node.add_child(c)
                node.left_remain = a.left_remain - cost
                node.right_remain = c.right_remain
                if c.single_atom:
                    node.right_remain = c.right_remain - cost
                assert node.left_remain >= 0
                assert node.right_remain >= 0
                node.single_atom = False
        elif node.symbol == 'aliphatic_organic' or node.symbol == 'aromatic_organic':
            min_valence = int(left_conn) + int(right_conn)
            if len(self.open_rings) and is_last:
                min_valence += 1
            candidates = []
            atom_types = avail_atoms[node.symbol]
            for i in range(len(atom_types)):
                a = atom_types[i]
                if atom_valence[a] >= min_valence:
                    if hasattr(node, 'banned_set') and a in node.banned_set:
                        continue
                    candidates.append(i)

            yield [self.get_mask_rule(node, candidates), FETCH_TOKEN]
            rule = self.rand_rule(node, candidates)
            
            
            a = self.get_node(node, atom_types[rule], 0)
            assert atom_valence[a.symbol] >= min_valence
            node.add_child(a)
            node.left_remain = atom_valence[a.symbol] - min_valence
            node.right_remain = atom_valence[a.symbol] - min_valence
            node.single_atom = True
            node.atom_pos = self.atom_num
            if node.symbol == 'aromatic_organic':
                node.is_aromatic = True
            else:
                node.is_aromatic = False
            self.atom_num += 1
        elif node.symbol == 'bond':
            candidates = []
            assert cap_remain            
            rr = range(len(bond_types))
            if hasattr(node, 'allowed'):
                rr = node.allowed
            for i in rr:
                b = bond_types[i]
                if bond_valence[b] <= cap_remain:
                    candidates.append(i)

            yield [self.get_mask_rule(node, candidates), FETCH_TOKEN]
            rule = self.rand_rule(node, candidates)
            
            
            b = self.get_node(node, bond_types[rule], 0)
            node.add_child(b)
        elif node.symbol == 'branched_atom':
            a = self.get_node(node, 'atom', 0)


            # self.tree_generator(a, left_conn, right_conn, is_last=is_last)
            for value in self.tree_generator(a, left_conn, right_conn, is_last=is_last):
                # If value is meant to be return, continue execution with it
                if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                    # In this case do nothing
                    continue
                else:
                    yield value


            node.atom_pos = a.atom_pos
            node.is_aromatic = a.is_aromatic
            node.add_child(a)

            candidates = set([0, 1, 2, 3])
            remain = int(a.left_remain)
            
            if len(self.open_rings) and is_last:
                remain += 1
                candidates.remove(0)
                pre_idx = node.get_pre()
                if self.maximum_match(pre_idx, remain) < len(self.open_rings):
                    candidates.remove(2)
                if remain < 2:
                    candidates.remove(3)
            else:
                if remain < 2:
                    candidates.remove(3)
                if remain < 1:
                    candidates.remove(2)
                    candidates.remove(1)
                if len(self.open_rings) == 0 and is_last:
                    if 2 in candidates:
                        candidates.remove(2)
                pre_idx = node.get_pre()
                if self.maximum_match(pre_idx, remain) == 0 and len(self.open_rings) == MAX_NESTED_BONDS:
                    assert not is_last
                    if 2 in candidates:
                        candidates.remove(2)
                    if 3 in candidates:
                        candidates.remove(3)
                if self.maximum_match(pre_idx, remain - 1) == 0 and len(self.open_rings) == MAX_NESTED_BONDS:
                    assert not is_last
                    if 3 in candidates:
                        candidates.remove(3)

            yield [self.get_mask_rule(node, list(candidates)), FETCH_TOKEN]
            rule = self.rand_rule(node, list(candidates))

            if rule > 1: # branched_atom -> atom RB | atom RB BB                        
                r = self.get_node(node, 'RB', 1)
                if rule == 2 and is_last:
                    r.task = True
                
                # remain = self.tree_generator(r, cap_remain=remain - (rule == 3), is_last=is_last)
                for value in self.tree_generator(r, cap_remain=remain - (rule == 3), is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        remain = value[0]
                    else:
                        yield value


                remain += (rule == 3)
                node.add_child(r)
                        
            node.left_remain = remain
            if rule % 2 == 1: # branched_atom -> atom BB | atom RB BB
                assert remain > 0
                b = self.get_node(node, 'BB', -1)
                b.pre_node = a.atom_pos
                assert b.pre_node is not None


                # node.left_remain = self.tree_generator(b, cap_remain = remain, is_last=is_last)
                for value in self.tree_generator(b, cap_remain = remain, is_last=is_last):
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        node.left_remain = value[0]
                    else:
                        yield value
                
                
                node.add_child(b)

            node.right_remain = node.left_remain
            node.single_atom = True
        elif node.symbol == 'RB':
            assert cap_remain
            b = self.get_node(node, 'ringbond', 0)
            b.task = node.task

            # cap_remain = self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
            for value in self.tree_generator(b, cap_remain=cap_remain, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap_remain = value[0]
                    else:
                        yield value
         
            node.add_child(b)
            
            candidates = []
            if node.task:
                candidates = [ int(len(self.open_rings) > 0) ]
            else:
                candidates = [0]
                pre_idx = node.get_pre()
                if cap_remain > 0 and not (self.maximum_match(pre_idx, cap_remain) == 0 and len(self.open_rings) == MAX_NESTED_BONDS):
                    candidates.append(1)
            
            yield [self.get_mask_rule(node, candidates), FETCH_TOKEN]
            rule = self.rand_rule(node, candidates)


            if rule == 1: # RB -> ringbond RB                
                assert cap_remain > 0
                r = self.get_node(node, 'RB', 1)
                r.task = node.task


                # cap_remain = self.tree_generator(r, cap_remain = cap_remain, is_last=is_last)
                for value in self.tree_generator(r, cap_remain = cap_remain, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap_remain = value[0]
                    else:
                        yield value


                node.add_child(r)
        elif node.symbol == 'BB':
            b = self.get_node(node, 'branch', 0)
            candidates = [0]
            assert cap_remain > 0
            if cap_remain > 1:
                candidates.append(1)
            
            yield [self.get_mask_rule(node, candidates), FETCH_TOKEN]
            rule = self.rand_rule(node, candidates)

            if rule == 1: # BB -> branch BB


                # rest = self.tree_generator(b, cap_remain=cap_remain - 1, is_last=False)
                for value in self.tree_generator(b, cap_remain=cap_remain - 1, is_last=False):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        rest = value[0]
                    else:
                        yield value
                
                
                node.add_child(b)
                bb = self.get_node(node, 'BB', 1)


                # rest = self.tree_generator(bb, cap_remain=rest + 1, is_last=is_last)
                for value in self.tree_generator(bb, cap_remain=rest + 1, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        rest = value[0]
                    else:
                        yield value                
                
                
                node.add_child(bb)
            else:
                # rest = self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                for value in self.tree_generator(b, cap_remain=cap_remain, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        rest = value[0]
                    else:
                        yield value

                node.add_child(b)

            cap_remain = rest
        elif node.symbol == 'ringbond':            
            pre_idx = node.get_pre()
            mm = self.maximum_match(pre_idx, cap_remain)
            if node.task:
                assert mm > 0 and mm >= len(self.open_rings)
            
            candidates = []
            # whether to match bond
            if mm > 0 and len(self.open_rings):
                for r in self.open_rings:
                    if self.ring_valid(r, pre_idx, cap_remain):
                        candidates.append(r)
            # whether to create bond
            if mm == 0 or (not node.task and len(self.open_rings) < MAX_NESTED_BONDS):
                assert len(self.open_rings) < MAX_NESTED_BONDS                
                candidates.append(MAX_NESTED_BONDS)
            
            yield [np.array(candidates) + TOTAL_NUM_RULES , FETCH_TOKEN]
            r = self.rand_att(node, candidates)

            bond_idx = r
            bond_type = '?'
            create = False
            if r == MAX_NESTED_BONDS: # create new bond
                for i in range(MAX_NESTED_BONDS):
                    if not i in self.open_rings and ((not i in self.sameatom_bonds[self.atom_num - 1]) or CONFIG.bondcompact):
                        bond_idx = i
                        create = True
                        break
                assert create
            else: # paired bond removed
                assert r in self.open_rings
                self.matched_bonds.add((self.open_rings[r].pos, self.atom_num - 1))
                bond_type = self.open_rings[r].b_type
                del self.open_rings[r]

            d = self.get_node(node, 'DIGIT', -1)
            r = self.get_node(d, '\'%d\'' %  (bond_idx + 1), 0)
            self.sameatom_bonds[self.atom_num - 1].add(bond_idx)

            d.add_child(r)
            node.add_child(d)

            if not create and bond_type is not None:
                yield [self.get_mask_rule(node, [1]), FETCH_TOKEN]
                rule = self.rand_rule(node, [1])
            else:
                yield [self.get_mask_rule(node), FETCH_TOKEN]
                rule = self.rand_rule(node)

            if rule == 1: # ringbond -> bond DIGIT
                b = self.get_node(node, 'bond', 0)
                if create:

                    # self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                    for value in self.tree_generator(b, cap_remain=cap_remain, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            continue
                        else:
                            yield value
                    
                    bond_type = b.children[0].symbol
                else:
                    assert cap_remain >= bond_valence[bond_type]
                    b.allowed = [0, 1, 2, 3, 4]


                    # self.tree_generator(b, cap_remain=cap_remain, is_last=is_last)
                    for value in self.tree_generator(b, cap_remain=cap_remain, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            continue
                        else:
                            yield value


                cap_remain -= bond_valence[b.children[0].symbol]
                node.add_child(b, 0)
            else:
                if bond_type == '?':
                    bond_type = None
                cap_remain -= 1

            if create:
                assert bond_type is None or bond_type != '?'
                self.open_rings[bond_idx] = RingBond(self.atom_num - 1, bond_type)
        elif node.symbol == 'branch':
            node.add_child(self.get_node(node, '\'(\'', 0))
            c = self.get_node(node, 'chain', -2)


            # self.tree_generator(c, left_conn=True, right_conn=False, is_last=is_last)
            for value in self.tree_generator(c, left_conn=True, right_conn=False, is_last=is_last):
                # If value is meant to be return, continue execution with it
                if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                    # In this case do nothing
                    continue
                else:
                    yield value
            
            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)

            cost = 1

            if rule == 1: # branch -> '(' bond chain ')'
                b = self.get_node(node, 'bond', 1)
                
                
                # self.tree_generator(b, cap_remain= min(cap_remain, c.left_remain + 1), is_last=is_last)
                for value in self.tree_generator(b, cap_remain= min(cap_remain, c.left_remain + 1), is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value

                
                cost = bond_valence[b.children[0].symbol]
                node.add_child(b)
            node.add_child(c)
            node.add_child(self.get_node(node, '\')\'', -1))
            cap_remain -= cost

         
         # this part of the code is modified due to the error when we do -1 - rule<2       
        elif node.symbol == 'BAI':

            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)


            if isinstance(rule, int):
                rule = torch.tensor(rule)

            if rule % 2 == 0: # BAI -> isotope xxx 
                i = self.get_node(node, 'isotope', 0)


                # self.tree_generator(i, is_last=is_last)
                for value in self.tree_generator(i, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value
                

                node.add_child(i)
            
            # This expects a tensor bool that can be typecast in this way
            # s = self.get_node(node, 'symbol',-1 - (rule < 2).int())
            # LOUIS: I've converted it to work with numpy since I'm passing numpy logits in
            s = self.get_node(node, 'symbol', -1 - int(rule < 2))
            s.banned_set = set(['\'B\''])


            # self.tree_generator(s, left_conn=left_conn, right_conn=right_conn, is_last=is_last)
            for value in self.tree_generator(s, left_conn=left_conn, right_conn=right_conn, is_last=is_last):
                # If value is meant to be return, continue execution with it
                if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                    # In this case do nothing
                    continue
                else:
                    yield value

            
            node.atom_pos = s.atom_pos
            node.add_child(s)

            cap = s.left_remain
            if rule <= 1: # BAI -> isotope aliphatic_organic BAC | aliphatic_organic BAC
                b = self.get_node(node, 'BAC', -1)


                # cap = self.tree_generator(b, cap_remain=cap, ref_symbol=s.children[0].symbol, is_last=is_last)
                for value in self.tree_generator(b, cap_remain=cap, ref_symbol=s.children[0].symbol, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap = value[0]
                    else:
                        yield value
                
                node.add_child(b)            
            node.left_remain = cap
            node.right_remain = cap
            node.single_atom = True
        elif node.symbol == 'BAC':

            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)

            if rule == 0 or rule == 2: # BAC -> chiral BAH | chiral
                c = self.get_node(node, 'chiral', 0)


                # self.tree_generator(c, is_last=is_last)
                for value in self.tree_generator(c, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value
                
                
                node.add_child(c)
            if rule <= 1: # BAC -> chiral BAH | BAH
                b = self.get_node(node, 'BAH', -1)


                # cap_remain = self.tree_generator(b, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last)
                for value in self.tree_generator(b, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap_remain = value[0]
                    else:
                        yield value
                

                node.add_child(b)
        elif node.symbol == 'BAH':
            if cap_remain == 0:

                yield [self.get_mask_rule(node, [0, 1]), FETCH_TOKEN]
                rule = self.rand_rule(node, [0, 1])

            else:
                yield [self.get_mask_rule(node), FETCH_TOKEN]
                rule = self.rand_rule(node)

            if rule <= 1: # BAH -> hcount charge | charge
                c = self.get_node(node, 'charge', -1)
                borrow = 0
                if cap_remain > 0 and rule == 0:
                    borrow = 1
                
                
                # cap_remain = self.tree_generator(c, cap_remain=cap_remain - borrow, ref_symbol=ref_symbol, is_last=is_last)
                for value in self.tree_generator(c, cap_remain=cap_remain - borrow, ref_symbol=ref_symbol, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap_remain = value[0]
                    else:
                        yield value


                cap_remain += borrow                
                node.add_child(c)
            if rule % 2 == 0: # BAH -> hcount charge | hcount
                assert cap_remain > 0
                hc = self.get_node(node, 'hcount', 0)


                # cap_remain = self.tree_generator(hc, cap_remain=cap_remain, is_last=is_last)
                for value in self.tree_generator(hc, cap_remain=cap_remain, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        cap_remain = value[0]
                    else:
                        yield value
                
                
                node.add_child(hc, 0)
        elif node.symbol == 'hcount':
            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)

            h = self.get_node(node, '\'H\'', 0)
            node.add_child(h)

            cost = 1
            if rule == 1: # hcount -> 'H' DIGIT
                d = self.get_node(node, 'DIGIT', -1)

                # self.tree_generator(d, cap_remain=cap_remain, is_last=is_last)
                for value in self.tree_generator(d, cap_remain=cap_remain, is_last=is_last):
                    # If value is meant to be return, continue execution with it
                    if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                        # In this case do nothing
                        continue
                    else:
                        yield value
                
                cost = int(d.children[0].symbol[1 : -1])
                node.add_child(d)
            cap_remain -= cost            
        elif node.symbol == 'charge':            
            if cap_remain == 0:

                yield [self.get_mask_rule(node, [2, 3]), FETCH_TOKEN]
                rule = self.rand_rule(node, [2, 3])

            else:
                yield [self.get_mask_rule(node), FETCH_TOKEN]
                rule = self.rand_rule(node)


            if rule <= 1: # charge -> '-' | '-' DIGIT
                m = self.get_node(node, '\'-\'', 0)
                node.add_child(m)
                cost = 1
                if rule == 1: # charge -> '-' DIGIT
                    d = self.get_node(node, 'DIGIT', -1)

                    # self.tree_generator(d, cap_remain=cap_remain, is_last=is_last)
                    for value in self.tree_generator(d, cap_remain=cap_remain, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            continue
                        else:
                            yield value
                    
                    cost = int(d.children[0].symbol[1 : -1])
                    node.add_child(d)
                cap_remain -= cost
            else: # charge -> '+' | '+' DIGIT
                p = self.get_node(node, '\'+\'', 0)
                node.add_child(p)                
                delta = 1
                if rule == 1: # charge -> '+' DIGIT
                    d1 = self.get_node(node, 'DIGIT', -1)


                    # self.tree_generator(d1, is_last=is_last)
                    for value in self.tree_generator(d1, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            continue
                        else:
                            yield value
                    

                    delta = int(d1.children[0].symbol[1 : -1])
                    node.add_child(d1)
                cap_remain += delta            
            assert ref_symbol is not None and ref_symbol != '\'B\''
        elif node.symbol == 'DIGIT':
            if cap_remain is None or cap_remain > len(prod[node.symbol]):

                yield [self.get_mask_rule(node), FETCH_TOKEN]
                rule = self.rand_rule(node)

            else:
                yield [self.get_mask_rule(node, range(cap_remain)), FETCH_TOKEN]
                rule = self.rand_rule(node, range(cap_remain))
                

            d = self.get_node(node, '\'%d\'' % (rule + 1), 0)
            node.add_child(d)
        else:
            assert node.symbol in ['smiles', 'atom', 'bracket_atom', 'isotope', 'chiral', 'symbol']
    
            yield [self.get_mask_rule(node), FETCH_TOKEN]
            rule = self.rand_rule(node)
    
            p = prod[node.symbol][rule]

            for i in range(len(p)):
                c = self.get_node(node, p[i], i)
                if not p[i][0] == '\'': # non-terminal


                    # t = self.tree_generator(c, left_conn, right_conn, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last)
                    for value in self.tree_generator(c, left_conn, right_conn, cap_remain=cap_remain, ref_symbol=ref_symbol, is_last=is_last):
                        # If value is meant to be return, continue execution with it
                        if isinstance(value, list) and value[1] == RETURN_TOKEN: # Short circuting
                            # In this case do nothing
                            t = value[0]
                        else:
                            yield value
                    
                    
                    node.left_remain = c.left_remain
                    node.right_remain = c.right_remain
                    node.single_atom = c.single_atom                    
                    node.atom_pos = c.atom_pos
                    node.is_aromatic = c.is_aromatic
                    if t >= 0:
                        cap_remain = t

                node.add_child(c)                
        
        # So cap remain here equals -1, but it should be None
        if cap_remain is not None: # Error: Get cap remain -1 here
            assert cap_remain >= 0
            yield [cap_remain, RETURN_TOKEN]
        else:
            yield [-1, RETURN_TOKEN]
        # Modified so that there's an else clause, otherwise it will 'return' twice!

    def decode(self, node, walker):
        self.walker = walker
        self.walker.reset()
        self.reset_state()
        yield from self.tree_generator(node, is_last=True)

def create_sequential_tree_decoder():
    fname = CONFIG.grammar_file.split('/')[-1]
    tree_decoder = AttMolGraphDecoderGenerator()
    return tree_decoder

if __name__ == '__main__':
    pass