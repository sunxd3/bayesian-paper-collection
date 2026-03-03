![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.15129141.svg)

Nathaniel Linden-Santangeli

University of California San Diego
2024

# Repository for: *Increasing certainty in systems biology models using Bayesian multimodel inference*
Authors: Nathaniel Linden-Santangeli, Jin Zhang, Boris Kramer, Padmini Rangamani

Date: June 2024

This repository provides all code and data that is necessary to reproduce all results shown in the manuscript. 
 <!-- available at [URL GOES HERE]. -->

## Instructions to reproduce results and figures
Source code is located in the `src/MAPK/` directory and results and data are located in the `results/` directory.

The excel file `source_data.xlsx` contains all results reported in the figures and tables of the manuscript.
Each associated script has code to generate the figure panels and tables in the manuscript and export the source data to the `results/` directory.

#### Local Identifiability analysis
The Julia code to run identifiability analysis for each model is located in `src/MAPK/identifiability/`. Each script runs the analysis for every model independently.

##### **Supplemental Figure S1** (Global sensitivity analysis)

1. Run `src/MAPK/gsa/gsa_sampling.sh` to generate samples of model outputs.
2. Run `src/MAPK/gsa/gsa_analyze.py` to perform Morris analysis and make plots.

#### Synthetic data generation and Keyes et al. *eLife*. 2020. data normalization.

All synthetic data generation and preprocessing occurs in `src/MAPK/process_data.ipynb`.

#### **Figure 3 and Supplemental Figures S2, S3** (Keyes et al. *eLife*. 2020. data inference/MMI)

0. (optional to rerun SMC) Run `src/MAPK/param_est/Keyes_CYTO_inference.sh` and `src/MAPK/param_est/Keyes_PM_inference.sh` without the `--skip_sample` flag on each call. Rerunning with this flag enabled regenerates posterior predictive samples from previous SMC results
1. Run `src/MAPK/param_est/Keyes_analyze.py` and `src/MAPK/param_est/Keyes_data_len_analyze.py` to plot posterior predictive trajectories
2. Run `src/MAPK/multimodel_inference/Keyes_MMI.ipynb` to perform MMI, run error analysis, and make plots
3. Run `src/MAPK/param_est/Keyes_data_params_compDiff_analyze.ipynb` to plot marginal eCDFs

#### **Supplemental Figures S4, S6** (Synthetic dose-response data)

*Figure S4*

0. (optional to rerun SMC) Run `src/MAPK/param_est/HF96_DR_inference.sh` without the `--skip_sample` flag on each call. Rerunning with this flag enabled regenerates posterior predictive samples from previous SMC results
1. Run `src/MAPK/param_est/HF96_DR_analyze.py` to plot posteriors
2. Run `src/MAPK/multimodel_inference/HF96_DR_MMI.ipynb` to perform MMI, run error analysis, and make plots

*Figure S6*
1. Run `src/MAPK/multimodel_inference/HF95_DR_model_perturb.ipynb` for panels A-E
2. Run `src/MAPK/multimodel_inference/HF95_DR_model_combinatorics.ipynb` for panels F,G

<!-- #### **Supplemental Figures S3, S4, and S6** (Synthetic trajectory data)

*Supplemental Figure S3, S4*

0. (optional to rerun SMC) Run `src/MAPK/param_est/HF96_traj_inference.sh` without the `--skip_sample` flag on each call. Rerunning with this flag enabled regenerates posterior predictive samples from previous SMC results
1. Run `src/MAPK/param_est/HF96_traj_analyze.py` to plot posterior predictive trajectories
2. Run `src/MAPK/multimodel_inference/HF96_traj_MMI.ipynb` to perform MMI, run error analysis, and make plots

*Supplemental Figure S6*
1. Run `src/MAPK/multimodel_inference/HF95_traj_model_perturb.ipynb` for panels X, Y, Z
2. Run `src/MAPK/multimodel_inference/HF95_traj_model_combinatorics.ipynb` for panels X, Y, Z -->


#### **Figure 4 and Supplemental Figure S7 and S8** (Keyes et al.  *eLife*. 2020. data shortening and quality reduction

*data shortening*

0. (optional to rerun SMC) Run `src/MAPK/param_est/Keyes_CYTO_10min_inference.sh`, `src/MAPK/param_est/Keyes_CYTO_20min_inference.sh`, `src/MAPK/param_est/Keyes_CYTO_30min_inference.sh`, `src/MAPK/param_est/Keyes_PM-10min_inference.sh`, `src/MAPK/param_est/Keyes_PM-20min_inference.sh`, and `src/MAPK/param_est/Keyes_PM-30min_inference.sh` without the `--skip_sample` flag on each call. Rerunning with this flag enabled regenerates posterior predictive samples from previous SMC results
1. Run `src/MAPK/param_est/Keyes_data_len_analyze.py` and `src/MAPK/param_est/Keyes_analyze.py` to plot posterior predictive trajectories
2. Run `src/MAPK/multimodel_inference/Keyes_MMI.ipynb` to perform MMI, run error analysis, and make plots

*data quality reduction*

1. [Requires SMC rerun--possible long runtime] Run `src/MAPK/param_est/Keyes_data_quality_inference.py`
2. Run `src/MAPK/multimodel_inference/Keyes_data_quality_MMI.ipynb`

#### **Figure 5 and Supplemental Figure S10** (Keyes et al.  *eLife*. 2020. Rap1/ERK negative feedback modifications)

0. (optional to rerun SMC) Run `src/MAPK/param_est/Keyes_Rap1_negFeed_inference.sh` without the `--skip_sample` flag on each call. Rerunning with this flag enabled regenerates posterior predictive samples from previous SMC results
1. Run `src/MAPK/param_est/Keyes_Rap1_negFeed_analyze.py` to plot posterior predictive trajectories
2. Run `src/MAPK/multimodel_inference/Keyes_Rap1_negFeed_MMI.ipynb` to perform MMI, run error analysis, and make plots

#### PSIS-LOO-CV vs. LOO-CV

The notebook `src/MAPK/LOOCV_analyze.ipynb` compares PSIS-LOO-CV to LOO-CV.

## Dependencies

### Julia

**Julia version 1.10.0** is used for local identifiability analysis.

##### Packages
- `StructuralIdentifiability.jl` - v0.5.6
- `DifferentialEquations.jl` - v7.10.0 

### Python

**Python version 3.11.6** is used for all other analysis.

##### Packages
- `Jax` - 0.4.25 (CPU computing)
- `Jaxlib` - 0.4.20 (CPU computing)
- `numpy` - 1.26.1
- `pandas` - 2.1.2
- `matplotlib` - 3.8.1
- `seaborn` - 0.13.0
- `PyMC` - 5.10.4
- `numpyro` - 0.13.2
- `pytensor` - 2.18.6
- `ArViz` - 0.16.1
- `Preliz` - 0.3.6
- `Diffrax` - 0.5.0
- `Optimistix` - 0.0.6
- `Lineax` - 0.0.4
- `tqdm` - 4.66.1
- `func_timeout` - 4.3.5 
