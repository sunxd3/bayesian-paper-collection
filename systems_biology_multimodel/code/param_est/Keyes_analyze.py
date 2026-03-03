import arviz as az
import pandas as pd
import json
import os

import numpy as np
import diffrax
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import sys
from scipy.stats import mode
from tqdm import tqdm


sys.path.append("../models/")
from huang_ferrell_1996 import *
from kholodenko_2000 import *
from levchenko_2000 import *
from hornberg_2005 import *
from birtwistle_2007 import *
from orton_2009 import *
from vonKriegsheim_2009 import *
from shin_2014 import *
from ryu_2015 import *
from kochanczyk_2017 import *

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, '../')
from utils import *

rng = np.random.default_rng(seed=1234)


################ LOAD in DATA ################
savedir = '../../../results/MAPK/param_est/Keyes_2020_data/'

# load in the model info 
model_info = json.load(open('model_info.json', 'r'))
model_names = list(model_info.keys())

# model_names.remove('vonKriegsheim_2009')
display_names = [model_info[model]['display_name'] for model in model_names]

idata = {'CYTO':{},'PM':{}}
posterior_samples = {'CYTO':{},'PM':{}}
sample_times = {'CYTO':{},'PM':{}}

for model in model_names:
    for compartment in ['CYTO','PM']:
        idata[compartment][model], _, sample_times[compartment][model] = load_smc_samples_to_idata(savedir+compartment+'/' + model + '/' + model +'_smc_samples.json', sample_time=True)
        posterior_samples[compartment][model] = np.load(savedir+compartment+'/' + model + '/' + model +'_posterior_predictive_samples.npy')

# # shin has 4000 samples so downsample to 2000
# idxs = rng.choice(np.arange(4000), size=2000, replace=False)
# posterior_samples['shin_2014'] = posterior_samples['shin_2014'][idxs,:,:]

# load in the training data
dat = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
       'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
dat['CYTO']['inputs'], dat['CYTO']['data'], dat['CYTO']['data_std'], \
    dat['CYTO']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-CYTO.json', data_std=True, time=True)
dat['PM']['inputs'], dat['PM']['data'], dat['PM']['data_std'], \
    dat['PM']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-PM.json', data_std=True, time=True)
data_time_to_mins = 60

# set up a color palette
# this is the ColorBrewer purple-green with 11 colors + three greys https://colorbrewer2.org/#type=diverging&scheme=PRGn&n=11
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b','#363737','#929591','#d8dcd6']
# this one gets to 10 colors by removing the darkest green
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#363737','#929591','#d8dcd6']
orange = '#de8f05'

colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837']

colors = get_color_pallette()
colors.remove(colors[-4]) # remove the second to last color, because we only have 10 models

################ Write sampling times to a file ################
with open(savedir + 'SMC_runtimes.txt', 'w') as f:
    for model in model_names:
        for compartment in ['CYTO','PM']:
            f.write(f'{model},{compartment}: {sample_times[compartment][model]/3600} hr\n')

################ Make pretty posterior predictive trajectories ################
SAM40_post_pred = {'CYTO':{},'PM':{}}            

skip_idxs = []
for idx, model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        for compartment in ['CYTO','PM']:
            print('plotting', model, 'in compartment', compartment)
            plot_posterior_trajectories(posterior_samples[compartment][model], 
                                        dat[compartment]['data'], dat[compartment]['data_std'], 
                                        dat[compartment]['times'], colors[idx], 
                                            dat[compartment]['inputs'], savedir+compartment+'/' + model + '/', model, data_time_to_mins=60,
                                            width=1., height=0.5, data_downsample=10,
                                            ylim=[[0.0, 1.5]],
                                            y_ticks=[[0.0, 1.0]],
                                            labels=False)
            
            # compute SAM40 post-pred predictions
            idx_40_min = -1
            SAM40_preds = np.apply_along_axis(sustained_activity_metric, 1, 
                                              posterior_samples[compartment][model], 
                                              idx_40_min)
            SAM40_post_pred[compartment][model] = list(SAM40_preds)

# save SAM40 predictions
with open(savedir + 'SAM40_post_pred.json', 'w') as f:
    json.dump(SAM40_post_pred, f)   

################ Make posterior trajectories and compute errors ################
# First we predict the entire data with posterior samples
# Then plot the data
# then compute relative training error and testing errors
n_traj = 400

plotting_params = [False,False,False,False]

errors = {'CYTO':{ 'RMSE':{}, 'rel_err':{}, 'RMSE_postpred':{}, 
                  'rel_err_postpred':{},'RMSE_final_10_min':{},'rel_err_final_10_min':{}},
        'PM':{'RMSE':{}, 'rel_err':{}, 'RMSE_postpred':{}, 'rel_err_postpred':{}, 
              'RMSE_final_10_min':{}, 'rel_err_final_10_min':{}}}

uncertainty = {'CYTO':{'cred95':{}, 'std':{}, 'cred95_postpred':{},
                       'std_postpred':{},'cred95_final_10_min':{}, 'std_final_10_min':{}},
                'PM':{ 'cred95':{}, 'std':{}, 'cred95_postpred':{}, 'std_postpred':{},
            'cred95_final_10_min':{}, 'std_final_10_min':{}}}

SAM40_pred = {'CYTO':{},'PM':{}}

skip_idxs = []
for idx,model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        for compartment in ['CYTO','PM']:
            this_model_info = model_info[model]

            plot_p = plotting_params

            # run posterior predictions if they do not already exist
            if not os.path.exists(savedir+compartment+'/' + model + '/traj_predict.npy'):
                # predict trajectories
                traj = predict_traj_response(model, idata[compartment][model], dat[compartment]['inputs'],
                                            dat[compartment]['times'], this_model_info['input_state'], this_model_info['ERK_states'],
                                            float(this_model_info['time_conversion']),
                                                    EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']),
                                                    nsamples=n_traj)
                traj = np.squeeze(traj)
                # save
                np.save(savedir+compartment+'/' + model + '/traj_predict.npy', traj)
            else:
                 traj = np.load(savedir+compartment+'/' + model + '/traj_predict.npy')
            
            
            # plot
            plot_posterior_trajectories(traj, dat[compartment]['data'], dat[compartment]['data_std'], 
                                        dat[compartment]['times'], colors[idx], 
                                        dat[compartment]['inputs'], savedir+compartment+'/' + model + '/', model, data_time_to_mins=60,
                                        width=1., height=0.5, 
                                        data_downsample=10,
                                        ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
                                        y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                                        fname='_pred_traj_', labels=False, train_times=dat[compartment]['times']/data_time_to_mins)
            plt.close('all')

            
            post_df = pd.DataFrame({'Time (min)': dat[compartment]['times']/data_time_to_mins, 
                                    'Mean Response': np.mean(traj, axis=0), 
                                    '2.5th':np.percentile(traj, 2.5, axis=0), 
                                    '97.5th':np.percentile(traj, 97.5, axis=0)})
            post_df.to_csv(savedir+compartment+'/' + model + '/' + model + '_' + compartment 
                           + '_posterior_data_summary.csv', index=False)
            # compute training error and testing error with RMSE and relative error
            # error posterior samples
            RMSE = np.sqrt(np.nanmean((np.nanmean(traj,axis=0) - dat[compartment]['data'])**2))
            rel_err = np.linalg.norm(np.nanmean( 
                traj,axis=0) - dat[compartment]['data'])/np.linalg.norm(
                dat[compartment]['data'])
            cred95 = np.nanmean(np.squeeze(np.diff(np.nanquantile(traj, [0.025, 0.975], axis=0),axis=0)))
            std = np.nanmean(np.nanstd(traj, axis=0))

            # error posterior predictive samples
            RMSE_postpred = np.sqrt(np.nanmean((np.nanmean(
                posterior_samples[compartment][model],axis=0) - \
                dat[compartment]['data'])**2))
            rel_err_postpred = np.linalg.norm(np.nanmean(
                posterior_samples[compartment][model],axis=0) - \
                dat[compartment]['data'])/np.linalg.norm(dat[compartment]['data'])
            cred95_postpred = np.nanmean(np.squeeze(np.diff(np.nanquantile(
                posterior_samples[compartment][model], [0.025, 0.975], axis=0),axis=0)))
            std_postpred = np.nanmean(np.nanstd(
                posterior_samples[compartment][model], axis=0))

            # error on final 10 minutes
            idx_30_min = np.argmin(abs((dat[compartment]['times']/data_time_to_mins)-30))
            dat_final_10_min = dat[compartment]['data'][idx_30_min:]
            pred_final_10_min = traj[:,idx_30_min:]
            RMSE_final_10_min = np.sqrt(np.nanmean((np.nanmean(
                pred_final_10_min,axis=0) - dat_final_10_min)**2))
            rel_err_final_10_min = np.linalg.norm(np.nanmean(
                pred_final_10_min,axis=0) - dat_final_10_min)/np.linalg.norm(dat_final_10_min)
            cred95_final_10_min = np.nanmean(np.squeeze(np.diff(np.nanquantile(
                pred_final_10_min, [0.025, 0.975], axis=0),axis=0)))
            std_final_10_min = np.nanmean(np.nanstd(pred_final_10_min, axis=0))
            
            errors[compartment]['RMSE'][model] = RMSE
            errors[compartment]['rel_err'][model] = rel_err
            errors[compartment]['RMSE_postpred'][model] = RMSE_postpred
            errors[compartment]['rel_err_postpred'][model] = rel_err_postpred
            errors[compartment]['RMSE_final_10_min'][model] = RMSE_final_10_min
            errors[compartment]['rel_err_final_10_min'][model] = rel_err_final_10_min

            uncertainty[compartment]['cred95'][model] = cred95
            uncertainty[compartment]['std'][model] = std
            uncertainty[compartment]['cred95_postpred'][model] = cred95_postpred
            uncertainty[compartment]['std_postpred'][model] = std_postpred
            uncertainty[compartment]['cred95_final_10_min'][model] = cred95_final_10_min
            uncertainty[compartment]['std_final_10_min'][model] = std_final_10_min

            # compute SAM40 predictions
            idx_40_min = -1
            SAM40_preds = np.apply_along_axis(sustained_activity_metric, 1, traj, idx_40_min)
            SAM40_pred[compartment][model] = list(SAM40_preds)

# save errors
with open(savedir + 'train_test_errors.json', 'w') as f:
    json.dump(errors, f)

# save uncertainty
with open(savedir + 'train_test_uncertainty.json', 'w') as f:
    json.dump(uncertainty, f)

 # save SAM40 predictions
with open(savedir + 'SAM40_predictions.json', 'w') as f:
    json.dump(SAM40_pred, f)