# Rethinking Molecular Design: Integrating Latent Variable and Auto-Regressive Models for Goal Directed Generation

This repository contains the code used for data preprocessing, model training and evaluation.

The latest version of the paper is available on [Arxiv](https://arxiv.org/pdf/2409.00046).

A more detailed presentation of the work, along with background and a review of the literature can be found in `/report/project_report.pdf`.


**Note on current state of directory:**

This directory is a work in progress. As of now only the evaluation and preprocessing code have been refactored. 

TODO:
- Add detailed running instructions for model training
- Fetch pretrained ZINC250k models from cluster and upload to Figshare
- Refactor model training directories
- Break up evaluation notebook into smaller ones
- Merge explicit and one-shot model training code into a single class

## Access Datasets and Pretrained Models

Pretrained versions of the models examined in the paper, along with the preprocessed versions of the QM9 and ZINC250k datasets have been made available via Figshare.

- [Processed ZINC250k dataset](https://figshare.com/articles/dataset/ZINC_250k_prepossessed_dataset_files_/28147751)
- [Processed QM9 dataset files](https://figshare.com/articles/dataset/Preprocessed_QM9_files/28147763)
- [Model files for trained versions of QM9 models](https://figshare.com/articles/software/QM9_Trained_Models/28147781) 
- Model files for trained versions of ZINC250k models | Comming soon ...

These should be downloaded and placed in `/data` and `/models` respectively.

## Preprocess Data (From Scratch)

You can recreate the data preprocessing yourself by performing the following steps:

### QM9
1. Raw QM9 data available from [here](https://figshare.com/articles/dataset/Data_for_6095_constitutional_isomers_of_C7H10O2/1057646?backTo=/collections/_/978904).
2. Extract .xyz folder in `data/qm9_36/`
3. Run preprocess_QM9.py with appropriate CMD line args


### ZINC250k
1. Download the ZINC250k .csv file from kaggle [here][https://www.kaggle.com/datasets/basu369victor/zinc250k].
2. Place `250k_rndm_zinc_drugs_clean_3.csv` in `/data/zinc250k_120/`
3. Run preprocess_ZINC250k.py with appropriate CMD line args


## Project Directory Structure

The project code is organised as follows:

```
├── data                        - Expected location of preprocessed versions of the datasets
│   ├── qm9_36
│   └── zinc250k_120
├── evaluation                  - Evaluation and plotting notebooks
├── explicit_vae                - Model code for explicit and teacher forcing versions of models 
│   ├── pretrained_models
│   └── reg_models
├── grammar                     - Smiles grammar specification for syntax directed models
├── lstm                        - Model code for surrogate model
│   └── gencond
├── models                      - Subdirectories for storing pretrained versions of models
│   ├── QM9
│   │   ├── explicit_vae
│   │   ├── explicit_vae_tf
│   │   ├── surrogate
│   │   └── vae
│   └── ZINC
│       └── surrogate
├── report                      - Contains extended project report
├── rnn                         - Depricated, will be removed soon
│   └── gencond
├── sd_lstm                     - LSTM model operating on attribute grammars, kept for future work
├── sd_vae                      - Modified implementation of SD-VAE, kept for future work
├── utils                       - Utility functions
└── vae                         - Model code for one-shot decoder models
    ├── pretrained_models
    └── reg_models
```