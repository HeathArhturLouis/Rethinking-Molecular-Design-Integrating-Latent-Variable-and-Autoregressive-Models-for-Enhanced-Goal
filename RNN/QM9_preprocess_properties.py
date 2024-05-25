#2019.8.16
# This file takes smiles train and test smiles string from QM9 data set and generates 
#two .npz file that contains smile strings and corresponding properties for traing and test set respectivly
from rdkit import Chem
import numpy as np
from pathlib import Path
from gencond.properties import PROPERTIES

from argparse import ArgumentParser


def process_properties(set):
	target = "./data/QM9/QM9_clean_smi_%s_smile.npz" % set

	if Path(target).exists():
	    print("Skipping %s data; file %s already exists" % (set, target))
	    return 

	smiles = np.load("./data/QM9/QM9_clean_smi_%s_smile.npy" % set )
	props = []
	for i in range(smiles.shape[0]):
	    props.append([prop(Chem.MolFromSmiles(smiles[i])) for prop in PROPERTIES.values()])
	    if len(props) % 10000 == 0:
	        print("processed %d strings" % len(props))
	        
	props = np.array(props)
	np.savez_compressed(target, smiles, props)       


if __name__ == '__main__':
    process_properties("train")
    process_properties("test")

