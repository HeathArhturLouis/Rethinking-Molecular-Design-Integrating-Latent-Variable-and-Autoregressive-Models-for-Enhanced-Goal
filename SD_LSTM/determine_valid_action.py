
import sys
import numpy as np

sys.path.append('../utils')
sys.path.append('../')

from nltk.grammar import Nonterminal, Production
import cfg_parser
from config import CONFIG


def one_hot_to_tree(one_hot_rules, grammar):
    # Initialize the root node
    root = Node(grammar.first_head.symbol())
    
    # Queue for breadth-first tree construction (start with the root node)
    node_queue = [(root, grammar.first_head)]

    for one_hot_rule in one_hot_rules:
        # Decode the one-hot rule to its rule index
        rule_index = np.argmax(one_hot_rule)

        if not node_queue:
            break  # No more nodes to expand

        # Get the current node and its corresponding nonterminal symbol from the queue
        current_node, current_symbol = node_queue.pop(0)

        # Find the applicable rule and its RHS symbols for the current nonterminal symbol
        for head, (start, end) in grammar.rule_ranges.items():
            if start <= rule_index < end:
                rule = grammar.head_to_rules[head][rule_index - start]  # Get the RHS of the rule
                current_node.rule_used = rule_index - start  # Set the rule used for the current node

                # Create child nodes for each symbol in the RHS of the rule and add them to the current node
                for symbol in rule:
                    child_node = Node(symbol)
                    current_node.add_child(child_node)
                    if isinstance(symbol, cfg_parser.Nonterminal):
                        node_queue.append((child_node, symbol))  # Add nonterminal child nodes to the queue for further expansion

                break

    return root


def determine_allowed_rules_mask(current_node, grammar):
    if current_node.is_created():
        # If the current node is fully expanded, look for the next expandable node in the tree
        # This part depends on how you traverse and keep track of expandable nodes in your tree
        pass  # Implement according to your tree traversal strategy

    applicable_rules = grammar.head_to_rules[current_node.symbol]
    
    # Create the allowed rules mask
    allowed_rules_mask = [0] * grammar.total_num_rules
    rule_start_index, _ = grammar.rule_ranges[current_node.symbol]
    for i, _ in enumerate(applicable_rules):
        allowed_rules_mask[rule_start_index + i] = 1

    return allowed_rules_mask


if __name__ == '__main__':

    grammar = cfg_parser.Grammar( '../' + CONFIG.grammar_file)

    all_derivations = np.load('../data/QM9/grammar_encodings.npy', allow_pickle=True)
    
    first_der = all_derivations[0]

    der_so_far = first_der[1:2]

    print(determine_allowed_rules_mask(der_so_far, grammar))
    print('Chosen:')
    print(first_der[2])

    # print(first_der)
    # print(all_derivations[0])
    # g
    