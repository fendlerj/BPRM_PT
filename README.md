# BPRM inference : MCMC with parallel tempering 

This code infers a Bayesian Profile Regression Mixture (BPRM) model with an MCMC algorithm including a parallel tempering scheme.

## Overview
This code is associated with the work **Considering parallel tempering and comparing post-treatment procedures in Bayesian Profile Regression Models for a survival outcome and correlated exposures**, Fendler J, Guihenneuc C and Ancelet A. 2025. For more details on the methodology, please consult the related article (https://doi.org/10.48550/arXiv.2512.23571).

A toy dataset (data.csv) is provided.

## Installation

### Requirements
- Python 3.10+
- R (≥ 3.5.0)

### Steps
```bash
git clone https://github.com/fendlerj/BPRM_PT.git
cd project
pip install -r requirements.txt
```
## Usage
To perform the inference using the MCMC algorithm run the file main_data_simu.py
```bash
python main_data_simu.py
```
To compute the similarity matrix run main_postproc.py
```bash
python main_postproc.py
```
To use highlight a clustering structure in the data based on the similarity matrix using the partitionning around the medoids post-processing, run PAM_postprocessing.R
```bash
Rscript PAM_postprocessing.py
```
Note : In all the previous file, the paths to find the code, the data and the results must be modified.

## License
MIT
