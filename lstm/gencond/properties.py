from guacamol.utils import descriptors
from collections import OrderedDict


PROPERTIES = OrderedDict(num_rotatable_bonds=descriptors.num_rotatable_bonds,
                         num_aromatic_rings=descriptors.num_aromatic_rings,
                         logP=descriptors.logP,
                         qed=descriptors.qed,
                         tpsa=descriptors.tpsa,
                         bertz=descriptors.bertz,
                         mol_weight=descriptors.mol_weight,
                         fluorine_count=descriptors.AtomCounter('F'),
                         num_rings=descriptors.num_rings)

TYPES = OrderedDict(zip(PROPERTIES.keys(), 
                        [int, int, float, float, float, float, float, int, int]))

