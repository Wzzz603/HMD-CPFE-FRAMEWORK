# HMD-CPFE-FRAMEWORK

Code and trained model parameters associated with the manuscript:

**A Hybrid Mechanism-Data Driven Framework for Crystal Plasticity Simulation: Synergistic Enhancement of Convergence and Efficiency via Deep Learning-Assisted Newton-Raphson under Large Increments**

This repository contains the author-developed components used for RVE preparation, machine-learning model training and baseline comparison, trained phase-specific model parameters, and Fortran-based machine-learning inference within the HMD-CPFE framework.

## Repository structure

```text
HMD-CPFE-FRAMEWORK/
├─ 1_rvebuild/
│  └─ dream3d.json
│
├─ 2_training/
│  ├─ geobuild/
│  │  ├─ 01_neibor_read.py
│  │  ├─ 02_generate_orientation_pools.py
│  │  ├─ 03_adaptive_basis_rve_builder_coverage_driven.py
│  │  └─ 04_inject_rve_info_to_inp_4groups_strict.py
│  │
│  └─ train/
│     ├─ 10_1_config.py
│     ├─ 10_2_h5_dataset.py
│     ├─ 10_3_normalizer.py
│     ├─ 10_4_models.py
│     ├─ 10_5_loss_and_metrics.py
│     ├─ 10_6_trainer.py
│     ├─ 10_7_train.py
│     ├─ 10_8_export_fortran_lstm.py
│     ├─ 10_9_train_linear_bcc_hcp.py
│     ├─ 10_9_train_ridge_bcc_hcp_FIXED.py
│     ├─ 10_9_train_lasso_bcc_hcp_FIXED.py
│     ├─ 10_9_train_polynomial2_bcc_hcp_FIXED.py
│     └─ 10_9_train_mlp_bcc_hcp.py
│
├─ 3_data/
│  └─ models/
│     ├─ BCC/
│     │  └─ BCC_best.txt
│     └─ HCP/
│        └─ HCP_best.txt
│
└─ 4_cpfe_ml/
   └─ mlmodule.f
```

## Contents

### 1. RVE construction

`1_rvebuild/dream3d.json` contains the DREAM.3D workflow configuration used in the RVE preparation procedure.

### 2. Training-data geometry preparation

The scripts in `2_training/geobuild/` support the preparation of RVE geometries used for machine-learning data generation, including periodic-neighbor information, crystallographic orientation-pool generation, adaptive basis-RVE generation, and injection of generated RVE information into Abaqus input files.

### 3. Machine-learning training

The scripts in `2_training/train/` provide the model-training workflow, including HDF5 dataset loading, train-only normalization, model definitions, loss and evaluation metrics, training/checkpoint management, LSTM training, and export of trained LSTM parameters for Fortran inference.

The repository also includes the baseline implementations used for comparison: Linear regression, Ridge regression, Lasso regression, second-order polynomial regression, and multilayer perceptron (MLP).

### 4. Trained phase-specific model parameters

`3_data/models/` contains the exported model parameters used by the Fortran inference module:

```text
BCC/BCC_best.txt
HCP/HCP_best.txt
```

These files correspond to the phase-separated BCC and HCP machine-learning models used in the HMD-CPFE framework.

### 5. CPFE-ML coupling

`4_cpfe_ml/mlmodule.f` contains the author-developed Fortran machine-learning module used to read the exported neural-network parameters and perform inference within the CPFE calculation workflow.

## Software requirements

The Python-side workflow requires a Python environment with the packages listed in `requirements.txt`.

The CPFE-side implementation was developed for Abaqus/Standard with a Fortran user-material workflow. Users should adapt paths, compiler settings, and Abaqus-specific configuration to their local installation.

## Path configuration

Some research scripts were originally developed in a local Windows environment and may contain local absolute paths. Before running the workflow on another system, update the relevant path variables in the configuration sections of the scripts.

## Data availability

The repository contains the author-developed source code and the trained phase-specific model parameter files used for deployment.

The complete raw and processed CPFE training datasets are not included in this GitHub repository because of their large size. They can be made available by the authors upon reasonable request.

## Third-party CPFE code

The underlying crystal-plasticity solver is based on the publicly available **OXFORD-UMAT** framework developed by the University of Oxford/Tarleton Group.

The original OXFORD-UMAT source code is **not redistributed in this repository**. Users should obtain the original CPFE implementation from its official source and comply with the corresponding terms of use and licensing conditions.

This repository is intended to provide the author-developed machine-learning, training, model-export, and CPFE-ML coupling components associated with the present work.

## Reproducibility scope

The public repository is intended to document and reproduce the author-developed data-driven portion of the framework, including:

1. preparation of RVE information for training-data generation;
2. training of LSTM and baseline machine-learning models;
3. export of trained BCC/HCP LSTM parameters;
4. Fortran-side loading and inference of the trained models.

Full end-to-end reproduction of the CPFE simulations additionally requires the external CPFE solver and an appropriate Abaqus/Fortran environment.

## Citation

If you use this repository, please cite the associated manuscript:

> Z. Wang, C. Duan, C. Wu, Y. Cai,  
> *A Hybrid Mechanism-Data Driven Framework for Crystal Plasticity Simulation: Synergistic Enhancement of Convergence and Efficiency via Deep Learning-Assisted Newton-Raphson under Large Increments*,  
> submitted to *Computational Materials Science*.

The bibliographic information will be updated after publication.

## Contact

For questions regarding the repository, code, or data availability, please contact the corresponding author through the contact information provided in the manuscript.
