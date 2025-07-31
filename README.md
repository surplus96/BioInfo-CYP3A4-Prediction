# BioInfo-CYP3A4-Contest

**Predictive modeling of CYP3A4 inhibition for molecule compounds.**

A fully reproducible pipeline for mining, featurizing and modeling CYP3A4 inhibitory activity of molecules.

## Table of Contents
1. Features
2. Project Layout
3. Quick Start
4. Data Pipeline
5. Feature Engineering
6. Model Training
7. Hyper-parameter Optimization
8. Prediction & Submission
9. Reproducing our Results
10. Requirements
11. Contributing
12. License
13. Acknowledgements

## Features
- Automated data mining from BindingDB & PubChem
- Clean merged dataset with IC₅₀ → pIC₅₀ conversion
- 2-D, 3-D and learned GNN descriptors
- Gradient-boosted tree and graph neural network models
- Simple ensembling that out-performs single models
- Modular design: each stage can be run or replaced independently

## Project Layout
```text
BioInfo-CYP3A4-Contest/
├── CYP3A4_data_miner/           # Raw data acquisition
├── CYP3A4_feature_engineering/  # Hand-crafted & 3-D descriptors
├── CYP3A4_model_train/          # GNN training & optimization
├── CYP3A4_tree_models/          # Classical tree learners
├── dataset/                     # Processed datasets & features
├── raw_data/                    # Original downloaded files
├── requirements.txt
└── README.md
```

## Quick Start
```bash
# Clone repository
$ git clone https://github.com/<your-org>/BioInfo-CYP3A4-Contest.git
$ cd BioInfo-CYP3A4-Contest

# Install Python dependencies
$ pip install -r requirements.txt

# End-to-end run (download → featurize → train → predict)
$ python preprocess_data.py                               # merge & clean
$ python CYP3A4_feature_engineering/generate_features.py  # 2-D
$ python CYP3A4_feature_engineering/generate_3d_features.py
$ python CYP3A4_model_train/train.py                      # GNN
$ python CYP3A4_tree_models/train_trees.py                # tree models
$ python CYP3A4_model_train/predict.py                    # final predictions
```

## Data Pipeline

The data pipeline is orchestrated by scripts under `CYP3A4_data_miner/` and `preprocess_data.py`.

1. **BindingDB / PubChem mining**  
   `bindingdb_miner.py` and `pubchem_miner.py` download raw SDF/CSV files that contain inhibition data against the CYP3A4 enzyme.
2. **Pre-processing & cleaning**  
   `preprocess_data.py` filters invalid measurements, removes duplicates and converts IC₅₀ to pIC₅₀ (stored in the `Inhibition` column).
3. **Dataset export**  
   Cleaned data are written to `dataset/cyp3a4_full_merged_dataset.csv` and split into train / test folds.

## Feature Engineering

| Module | Description |
| ------ | ----------- |
| `generate_features.py` | Calculates 2-D RDKit physico-chemical descriptors. |
| `generate_3d_features.py` | Generates 3-D descriptors such as WHIM & MOE. |
| `extract_gnn_features.py` | Obtains learned embeddings from a pretrained Chemprop GNN. |
| `select_features.py` | Optional feature selection with mutual information. |

All resulting matrices are saved inside `dataset/train_featured/`.

## Model Training

Two complementary model families are trained:

1. **Graph Neural Network (Chemprop)**  
   Scripts in `CYP3A4_model_train/` fine-tune the Chemprop model on canonical SMILES while optionally concatenating engineered features.
2. **Tree-based models**  
   `CYP3A4_tree_models/train_trees.py` fits XGBoost, LightGBM & CatBoost on the engineered features.

## Hyper-parameter Optimization

`CYP3A4_model_train/optimize.py` performs a lightweight Random Search over Chemprop hyper-parameters (depth, hidden size, dropout, etc.).  
Top-performing parameters are saved to `best_gnn_params.json`.

## Prediction & Submission

After training, run

```bash
python CYP3A4_model_train/predict.py                      # writes gnn_predictions.csv
python CYP3A4_tree_models/submission_final_ensembled.csv  # pre-computed ensemble
```

The final Kaggle submission file will be located in `submission_final_ensembled.csv`.

## Reproducing Our Results

We provide a deterministic split seed and fixed random seeds for NumPy, PyTorch and XGBoost so that results can be reproduced exactly.

```bash
export PYTHONHASHSEED=0
python CYP3A4_model_train/train.py --seed 42
```

## Requirements

* Python ≥ 3.9
* RDKit
* scikit-learn
* XGBoost, LightGBM, CatBoost
* PyTorch ≥ 1.13
* Chemprop ≥ 1.5

Full versions are listed in `requirements.txt`.

## Contributing

Pull requests are welcome!  Please open an issue first to discuss your proposed change.

## License

Distributed under the MIT License.  See `LICENSE` for more information.