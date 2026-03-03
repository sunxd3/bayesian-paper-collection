import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import seaborn as sns
import jax
import sys
from scipy.stats import mode
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '../')
from utils import *

rng = np.random.default_rng(seed=1234)


################ LOAD in DATA ################
savedir = '../../../results/MAPK/param_est/Keyes_2020_data_locDiffs/'

orton_cases_df = pd.read_csv('./orton_2009_loc_diff_models_params.csv', delimiter='.',
                            names=['case', 'diff_params'])
shin_cases_df = pd.read_csv('./shin_2014_loc_diff_models_params.csv', delimiter='.',
                            names=['case', 'diff_params'])
ryu_cases_df = pd.read_csv('./ryu_2015_loc_diff_models_params.csv', delimiter='.',
                            names=['case', 'diff_params'])

model_names = {'orton_2009':list(orton_cases_df['case'].values),
               'shin_2014':list(shin_cases_df['case'].values),
               'ryu_2015':list(ryu_cases_df['case'].values),}

# load in the training data
data_file = '../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'

inputs_CYTO, data_CYTO, data_std_CYTO, times_CYTO \
    = load_data_json(data_file+'CYTO_CYTOmax.json', data_std=True, time=True)
inputs_PM, data_PM, data_std_PM, times_PM \
    = load_data_json(data_file+'PM_PMmax.json', data_std=True, time=True)

data_file = '../../../results/MAPK/Keyes_et_al_2020-fig3-data1-v2-'
inputs_CYTO_RAP1i, data_CYTO_RAP1i, data_std_CYTO_RAP1i, \
    times_CYTO_RAP1i = load_data_json(data_file+'CYTO_RAP1inhib_CYTOmax.json', \
    data_std=True, time=True)
inputs_PM_RAP1i, data_PM_RAP1i, data_std_PM_RAP1i, times_PM_RAP1i \
    = load_data_json(data_file+'PM_RAP1inhib_PMmax.json', \
    data_std=True, time=True)

data_time_to_mins = 60

# data in each compartment are sampled at slightly different times, so we need to interpolate to align them
# use diffrax linear interpolation to get the MAPK activity at specific time point over 40mins
data_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_CYTO)
data_std_CYTO_interp = diffrax.LinearInterpolation(times_CYTO, data_std_CYTO)
data_PM_interp = diffrax.LinearInterpolation(times_PM, data_PM)
data_std_PM_interp = diffrax.LinearInterpolation(times_PM, data_std_PM)

data_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i, \
    data_CYTO_RAP1i)
data_std_CYTO_RAP1i_interp = diffrax.LinearInterpolation(times_CYTO_RAP1i,\
    data_std_CYTO_RAP1i)
data_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, data_PM_RAP1i)
data_std_PM_RAP1i_interp = diffrax.LinearInterpolation(times_PM_RAP1i, \
    data_std_PM_RAP1i)

min_time = np.round(np.min([times_CYTO[-1], times_PM[-1], times_CYTO_RAP1i[-1], times_PM_RAP1i[-1]]))
n_times = np.max([len(times_CYTO), len(times_PM), len(times_CYTO_RAP1i), len(times_PM_RAP1i)])
times = np.linspace(0, min_time, n_times)

# get data at standard times
data = {
    'CYTO':data_CYTO_interp.evaluate(times),
    'PM':data_PM_interp.evaluate(times),
    'CYTO_Rap1KD':data_CYTO_RAP1i_interp.evaluate(times),
    'PM_Rap1KD':data_PM_RAP1i_interp.evaluate(times)}

data_std = {
    'CYTO':data_std_CYTO_interp.evaluate(times),
    'PM':data_std_PM_interp.evaluate(times),
    'CYTO_Rap1KD':data_std_CYTO_RAP1i_interp.evaluate(times),
    'PM_Rap1KD':data_std_PM_RAP1i_interp.evaluate(times)}

# get standard color palette
n_models = len(model_names['orton_2009']) + len(model_names['shin_2014']) + len(model_names['ryu_2015'])
colors = get_color_pallette(n_colors=n_models, append_colors=[])

m_names = [model + '_' + submodel for model in model_names.keys() for submodel in model_names[model]]
color_dict = {m_names[i]:colors[i] for i in range(len(m_names))}

for idx, model in enumerate(model_names.keys()):
    for submodel in model_names[model]:
        name = savedir + model + '/' + submodel + '/'
        if model in ['shin_2014', 'ryu_2015']:
            model_ = model + '_Rap1'
        else:
            model_ = model
    
        idata, _ = load_smc_samples_to_idata(name + submodel + model_ + '_smc_samples.json')

        prediction_CYTO = np.squeeze(idata.posterior['prediction_CYTO'].values)
        prediction_CYTO = prediction_CYTO.reshape(-1, prediction_CYTO.shape[-1])
        prediction_PM = np.squeeze(idata.posterior['prediction_PM'].values)
        prediction_PM = prediction_PM.reshape(-1, prediction_PM.shape[-1])
        prediction_CYTO_Rap1KD = np.squeeze(idata.posterior['prediction_CYTO_Rap1KD'].values)
        prediction_CYTO_Rap1KD = prediction_CYTO_Rap1KD.reshape(-1, prediction_CYTO_Rap1KD.shape[-1])
        prediction_PM_Rap1KD = np.squeeze(idata.posterior['prediction_PM_Rap1KD'].values)
        prediction_PM_Rap1KD = prediction_PM_Rap1KD.reshape(-1, prediction_PM_Rap1KD.shape[-1])

        llike_samples = {
            'CYTO': prediction_CYTO,
            'PM': prediction_PM,
            'CYTO_Rap1KD': prediction_CYTO_Rap1KD,
            'PM_Rap1KD': prediction_PM_Rap1KD
        }

        for comp in ['CYTO', 'PM']:
            # plot the posterior predictive trajectories
            samples_ = np.stack([llike_samples[comp], llike_samples[comp+'_Rap1KD']])
            # add additional zeros to each sample for the ICs (this is just to make plotting easier, we know the ICs are zero)
            zer_col = np.zeros((samples_.shape[0], samples_.shape[1], 1))
            samples_ = np.concatenate([zer_col, samples_], axis=2)
            # reshape the samples_ martix so that it is n_traj x n_comp x n_times
            samples_ = np.swapaxes(samples_, 0, 1)

            data_ = np.stack([data[comp], data[comp+'_Rap1KD']])
            data_std_ = np.stack([data_std[comp], data_std[comp+'_Rap1KD']])
            data_std_nan = np.nan*np.ones_like(data_std_)

            plot_posterior_trajectories(samples_, data_, data_std_nan, times,
                color_dict[model+'_'+submodel], ['', '_Rap1KD'], 
                name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive', '',
                fname='',
                data_time_to_mins=60,
                width=1., height=0.5, 
                data_downsample=10,
                ylim=[[0.0, 1.5],[0.0, 1.5]],
                y_ticks=[[0.0, 1.0],[0.0, 1.0]],
                labels=False,)
            
            post_df = pd.DataFrame({'Time (min)': times, 
                                    'Mean Response': np.mean(samples_[:,0,:], axis=0), 
                                    '2.5th':np.percentile(samples_[:,0,:], 2.5, axis=0), 
                                    '97.5th':np.percentile(samples_[:,0,:], 97.5, axis=0)})
            post_df.to_csv(name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive.csv', index=False)
            post_df = pd.DataFrame({'Time (min)': times, 
                                    'Mean Response': np.mean(samples_[:,1,:], axis=0), 
                                    '2.5th':np.percentile(samples_[:,1,:], 2.5, axis=0), 
                                    '97.5th':np.percentile(samples_[:,1,:], 97.5, axis=0)})
            post_df.to_csv(name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive_Rap1KD.csv', index=False)

# plot posterior trajectories for Orton 2009 and Ryu 2015 with negFB KD
submodel = 'Rap1_negFB'
colors = {
    'orton_2009':(0.7423529411764707, 0.5282352941176469, 0.7015686274509803),
    'ryu_2015':(0.6891176470588235, 0.3857843137254902, 0.14617647058823535)
}

for model in ['orton_2009', 'ryu_2015']:
    name = savedir + model + '/' + submodel + '/'

    if model == 'ryu_2015':
        model_ = model + '_Rap1'
    else:
        model_ = model

    rap1KD_negFBKD_CYTO = np.load(name + submodel + model_ + '_CYTO_Rap1KD_posterior_negFeedbackKD_predictive_samples.npy')
    rap1KD_negFBKD_PM = np.load(name + submodel + model_ + '_PM_Rap1KD_posterior_negFeedbackKD_predictive_samples.npy')
    negFBKD_CYTO = np.load(name + submodel + model_ + '_CYTO_posterior_negFeedbackKD_predictive_samples.npy')
    negFBKD_PM = np.load(name + submodel + model_ + '_PM_posterior_negFeedbackKD_predictive_samples.npy')

    print(rap1KD_negFBKD_CYTO.shape, rap1KD_negFBKD_PM.shape, negFBKD_CYTO.shape, negFBKD_PM.shape)
    
    KD_samples = {
        'CYTO_negFBKD': negFBKD_CYTO,
        'PM_negFBKD': negFBKD_PM,
        'CYTO_Rap1KD_negFBKD': rap1KD_negFBKD_CYTO,
        'PM_Rap1KD_negFBKD': rap1KD_negFBKD_PM
    }

    for comp in ['CYTO', 'PM']:
        # plot the posterior predictive trajectories
        samples_ = np.stack([KD_samples[comp+'_negFBKD'], KD_samples[comp+'_Rap1KD_negFBKD']])
        # add additional zeros to each sample for the ICs (this is just to make plotting easier, we know the ICs are zero)
        zer_col = np.zeros((samples_.shape[0], samples_.shape[1], 1))
        samples_ = np.concatenate([zer_col, samples_], axis=2)
        # reshape the samples_ martix so that it is n_traj x n_comp x n_times
        samples_ = np.swapaxes(samples_, 0, 1)

        data_ = np.nan*np.ones_like(np.stack([data[comp], data[comp+'_Rap1KD']]))
        data_std_ = np.stack([data_std[comp], data_std[comp+'_Rap1KD']])
        data_std_nan = np.nan*np.ones_like(data_std_)

        plot_posterior_trajectories(samples_, data_, data_std_nan, times,
            colors[model], ['_negFBKD', '_Rap1KD_negFBKD'], 
            name + submodel + '_' + model_ + '_' + comp + '_post_sims', '',
            fname='',
            data_time_to_mins=60,
            width=1., height=0.5, 
            data_downsample=10,
            ylim=[[0.0, 1.5],[0.0, 1.5]],
            y_ticks=[[0.0, 1.0],[0.0, 1.0]],
            labels=False,)
        
        post_df = pd.DataFrame({'Time (min)': times, 
                                    'Mean Response': np.mean(samples_[:,0,:], axis=0), 
                                    '2.5th':np.percentile(samples_[:,0,:], 2.5, axis=0), 
                                    '97.5th':np.percentile(samples_[:,0,:], 97.5, axis=0)})
        post_df.to_csv(name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive_negFBKD.csv', index=False)
        post_df = pd.DataFrame({'Time (min)': times, 
                                'Mean Response': np.mean(samples_[:,1,:], axis=0), 
                                '2.5th':np.percentile(samples_[:,1,:], 2.5, axis=0), 
                                '97.5th':np.percentile(samples_[:,1,:], 97.5, axis=0)})
        post_df.to_csv(name + submodel + '_' + model_ + '_' + comp + '_posterior_predictive_Rap1KD_negFBKD.csv', index=False)
