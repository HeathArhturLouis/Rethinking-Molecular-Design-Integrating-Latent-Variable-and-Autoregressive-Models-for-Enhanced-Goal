'''
Global Configuration
'''
import os

class CONFIG:
    target_props = ['LogP'] # ['QED', 'SA', 'LogP']
    grammar_file = os.path.join('./grammar', 'mol_zinc.grammar')
    max_decode_steps = 100
    info_folder = './grammar/'
    skip_deter = 0 # Skip deterministic position
    bondcompact = 0 # Compact ringbond or not
    loss_type = 'big error loss perp' # TODO change back; Perplexity loss, other options is binary CE ('binary')