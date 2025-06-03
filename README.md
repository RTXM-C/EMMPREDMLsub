# EMMPREDMLsub

**EMMPREDMLsub** is a deep learning-based framework for predicting mRNA subcellular localization in a multi-label setting. It integrates ESM2 for powerful sequence representation and adopts the MMDO-MDPU resampling strategy to handle imbalanced datasets, improving prediction robustness and accuracy.
The online predictor can be found at http://www.emmpredmlsub.com/

## Repository Structure

- `main_classify.py`: Main program file, which can be called for sequence prediction.
- `model13.py`: Defines the deep learning model structure and parameters settings.
- `predict.py`: Script for making predictions using a trained model.
- `sample.txt`: A sample file containing mRNA sequences for inference.
- `model/`: Trained model files, including ExtraTrees、XGBoost、LightGBM、DeepLearning.

## Requirements

Ensure the following dependencies are installed, the version is for reference only:

- python 3.8
- bio 1.6.2
- fair-esm 2.0.0
- joblib 1.4.2
- lightgbm 4.6.0
- scikit-learn 1.3.2
- torch 2.4.1 
- torch-geometric 2.6.1
- xgboost 2.1.4

## Usage

### Clone the repository

```bash
git clone https://github.com/RTXM-C/EMMPREDMLsub.git
cd EMMPREDMLsub
```

### Predict using the trained model

```bash
python main_classify.py
```

## Input Format

Prepare your input file (`sample.txt`) with one mRNA sequence per line, make sure it is in fasta format. For example:

```
>000010000|#1
GCATCTAGT...CAAGTGTGAATTGC
```

## Citation

If you find this work useful in your research, please consider citing our work (citation info to be added when available).

## Contact

If you have questions or suggestions, please feel free to open issue or contact the author via GitHub, or contact me via email haoyueluo184@gmail.com.
