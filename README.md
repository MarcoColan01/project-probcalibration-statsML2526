# Calibration of Probabilities
This repository contains the project for the "Statistical Methods for Machine Learning" course (a.y. 2025/26), Master Degree in Computer Science at University of Milan. The author of the project is Marco Colangelo (ID 67045A).

## Abstract
This study evaluates and improves the calibration of the probabilities predicted by Logistic Regression and Random Forest, following the experimental methodology of Niculescu-Mizil and Caruana. Both models are trained on three binary classification datasets that differ in label noise, feature types and class balance (HIGGS, Covertype and Adult), and their predictions are post-processed with Platt scaling and isotonic regression, implemented from scratch and fitted on an independent calibration set. Calibration is assessed with reliability diagrams and prediction histograms, and quantified with accuracy, log-loss, Brier score and expected calibration error (ECE). 

The experiments show that Logistic Regression is well calibrated where the linear model fits the data and only mildly miscalibrated elsewhere, while Random Forest exhibits the sigmoid-shaped under-confidence described in the paper, strongest on the low-noise Covertype dataset.  Before calibration, Logistic Regression gives the best-calibrated probabilities on all datasets; after calibration, Random Forest becomes equally well calibrated on HIGGS and Covertype, where it gives the best probabilities, while Logistic Regression remains the best model on Adult. Isotonic regression corrects every observed distortion when enough calibration data is available, whereas Platt scaling is effective on sigmoid-shaped distortions but degrades well-calibrated models whose predictions concentrate near $0$ and $1$. Calibration leaves the accuracy essentially unchanged, and a learning-curve analysis confirms that Platt scaling needs only a few hundred calibration examples, while isotonic regression needs thousands.

Here is the PDF report for this project: [report.pdf](report/main.pdf)

## Repository structure
The code is organized into `.py` modules and `.ipynb` notebooks that utilize the `.py` modules for datasets operations and results generation.

```text
├── notebooks/
    ├── 1_datasets.ipynb                # Loading, preprocessing and split of datasets
    ├── 2_training_calibration.ipynb    # Training and calibration of models and plots generation
├── src/
│   ├── data.py                 # Loading, preprocessing and split of datasets
│   ├── calibration.py          # Calibration methods (Platt scaling and isotonic regression)
│   ├── metrics.py              # Evaluation metrics
│   └── plots.py                # Base code for plots generation
├── report/
│   ├── main.tex                # Source code for report
|   ├── references.bib          # Bibliography 
│   └── main.pdf                # Compiled report in PDF format
```

## How to reproduce results

1. Create a virtual environment and install required dependencies.
    ```bash
    python -m venv .venv 
    pip install -r requirements.txt
    ```
2. Run [`1_datasets.ipynb`](notebooks/1_datasets.ipynb) first for datasets loading, preprocessing and splits generation, then run [`2_training_calibration.ipynb`](notebooks/2_training_calibration.ipynb) to reproduce all the results.


