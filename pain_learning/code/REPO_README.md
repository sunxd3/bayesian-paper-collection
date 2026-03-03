# Code for the relevant analysis and results:
### "Statistical learning shapes pain perception and prediction independently of external cues"
Onysk, J., Whitefield, M., Gregory, N., Jain, M., Turner, G., Seymour, B., Mancini, F. (2024). eLife. https://doi.org/10.7554/eLife.90634.2

## 1 - data_collection

Contains the code for the psychophysical experiment (PsychToolBox), including the sequence generations scripts.

## 2 - preprocessing

Contains code that preprocesses behavioural data from PsychToolBox. This includes linear transformation of inputs, exporting data to stan readable format and plotting Supplement figures

## 3 - model_fit_analysis

Contains stan models used in the paper ('models/'), model fitting code ('fit_models_cs.R) (inlcuding HPC setup in 'hpc/'), initial analysis script for processing stan samples ('primary_analysis_cs.R'), as well as additional analyis scripts ('extra_analysis_cs.R', 'correlation_beh_model_cs.ipynb') that generate figures from the paper and supplement.

The following files exceed git repository size quota and can be found on OSF (https://osf.io/q7fvr/):
- model_fit_analysis/output/cs_results/sig_draws_cond.csv
- model_fit_analysis/output/cs_results/gr_draws_cond.csv

## 4 - model_recovery

Contains code that execute model and parameter recovery analysis ('mp_recovery.R'), including HPC setup ('hpc/'). The 'mp_rec_analyse.R' reproduces model and parameter recovery results from the supplement.

## 5 - Figures

Contain all the figures from the manuscript and the supplement.

---------------

Full dataset, inlucidng RStan RDS fit objects can be found on Zenodo: https://zenodo.org/records/11394627
