from collections import OrderedDict
from guacamol.utils import descriptors
import sascorer


# Maps property names -> (fun : RDKit MOL --> Property Score)
class PropertyCalculator:
    '''
    When instantiated computes a set of property descriptors of a RDKit Molecular representation
    '''
    PROP_FUNCTIONS = OrderedDict([
                ('LogP', descriptors.logP),
                ('QED', descriptors.qed),
                ('SA', sascorer.calculateScore)
            ])

    PROP_TYPES = OrderedDict([
                ('LogP', float),
                ('QED', float),
                ('SA', float)
            ])
    
    def __init__(self, prop_names = None):
        self.properties = prop_names

    def __call__(self, mol, prop_names=None):
        '''
        Inputs: mol : RDKit Molecule
                prop_names : List of property names strings for the properties from PROP_FUNCTIONS/TYPES to compute
                    If None will use initialized property list (self.properties)
        Outputs: List of properties of mol in the order of self.prop_names 
        '''
        if prop_names == None:
            return [(PropertyCalculator.PROP_FUNCTIONS[pr])(mol) for pr in self.properties]
        else:
            return [(PropertyCalculator.PROP_FUNCTIONS[pr])(mol) for pr in prop_names]
        
if __name__ == '__main__':
    '''
    TESTING CODE: TODO: Remove this 
    '''
    from rdkit import Chem
    smiles = 'CCO'
    props = ['LogP', 'SA', 'QED']

    mol = Chem.MolFromSmiles(smiles)
    pc = PropertyCalculator(['QED', 'LogP'])
    print(pc(mol))
    print(pc(mol, props))
