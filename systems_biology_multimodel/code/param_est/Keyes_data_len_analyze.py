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

# load in the training data
dat_full = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
       'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
dat_full['CYTO']['inputs'], dat_full['CYTO']['data'], dat_full['CYTO']['data_std'], \
    dat_full['CYTO']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-CYTO.json', data_std=True, time=True)
dat_full['PM']['inputs'], dat_full['PM']['data'], dat_full['PM']['data_std'], \
    dat_full['PM']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-PM.json', data_std=True, time=True)
data_time_to_mins = 60

# loop over different training data lengths
for train_len in [10, 20, 30]:
    ################ LOAD in DATA ################
    savedir = '../../../results/MAPK/param_est/Keyes_2020_data_'+str(train_len)+'min/'


    # load in the model info 
    model_info = json.load(open('model_info.json', 'r'))
    model_names = list(model_info.keys())
    
    display_names = [model_info[model]['display_name'] for model in model_names]

    idata = {'CYTO':{},'PM':{}}
    posterior_samples = {'CYTO':{},'PM':{}}
    sample_times = {'CYTO':{},'PM':{}}

    errors = {'CYTO':{
        'RMSE_train':{},
        'rel_err_train':{},
        'RMSE_test':{},
        'rel_err_test':{},
        'RMSE_full_data':{},
        'rel_err_full_data':{},
        'RMSE_final_10_min':{},
        'rel_err_final_10_min':{}
    },'PM':{
        'RMSE_train':{},
        'rel_err_train':{},
        'RMSE_test':{},
        'rel_err_test':{},
        'RMSE_full_data':{},
        'rel_err_full_data':{},
        'RMSE_final_10_min':{},
        'rel_err_final_10_min':{}
    }}

    uncertainty = {'CYTO':{
        'cred95_train':{},
        'std_train':{},
        'cred95_test':{},
        'std_test':{},
        'cred95_full_data':{},
        'std_full_data':{},
        'cred95_final_10_min':{},
        'std_final_10_min':{}
    },'PM':{
        'cred95_train':{},
        'std_train':{},
        'cred95_test':{},
        'std_test':{},
        'cred95_full_data':{},
        'std_full_data':{},
        'cred95_final_10_min':{},
        'std_final_10_min':{}
    }}

    SAM40_pred = {'CYTO':{},'PM':{}}

    for model in model_names:
        for compartment in ['CYTO','PM']:
            idata[compartment][model], _, sample_times[compartment][model] = load_smc_samples_to_idata(savedir+compartment+'/' + model + '/' + model +'_smc_samples.json', sample_time=True)
            posterior_samples[compartment][model] = np.load(savedir+compartment+'/' + model + '/' + model +'_posterior_predictive_samples.npy')

    # load in the training data
    dat = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
        'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
    dat['CYTO']['inputs'], dat['CYTO']['data'], dat['CYTO']['data_std'], \
        dat['CYTO']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-CYTO-'+str(train_len)+'min.json', data_std=True, time=True)
    dat['PM']['inputs'], dat['PM']['data'], dat['PM']['data_std'], \
        dat['PM']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-PM-'+str(train_len)+'min.json', data_std=True, time=True)
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
                                                width=1.1, height=0.5, data_downsample=10,
                                                ylim=[[0.0, 1.5]],
                                                y_ticks=[[0.0, 1.0]],
                                                labels=False, xlim=[0,60])
                plt.close('all')
            
    ################ Make predictions of the testing data ################
    # First we predict the entire data with posterior samples
    # Then plot the data
    # then compute relative training error and testing errors
    n_traj = 400

    plotting_params = [False,False,False,False]

    skip_idxs = []
    for idx,model in enumerate(model_names):
        if idx in skip_idxs:
            print('skipping', model)
            continue
        else:
            for compartment in ['CYTO','PM']:
                print('plotting', model)
                this_model_info = model_info[model]

                plot_p = plotting_params

                # # predict trajectories
                if not os.path.exists(savedir+compartment+'/' + model + '/traj_predict.npy'):
                    traj = predict_traj_response(model, idata[compartment][model], dat_full[compartment]['inputs'],
                                                dat_full[compartment]['times'], this_model_info['input_state'], this_model_info['ERK_states'],
                                                float(this_model_info['time_conversion']),
                                                        EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']),
                                                        nsamples=n_traj)
                    traj = np.squeeze(traj)
                    # save
                    np.save(savedir+compartment+'/' + model + '/traj_predict.npy', traj)
                else:
                    traj = np.load(savedir+compartment+'/' + model + '/traj_predict.npy')
               
                # plot
                plot_posterior_trajectories(traj, dat_full[compartment]['data'], dat_full[compartment]['data_std'], 
                                            dat_full[compartment]['times'], colors[idx], 
                                            dat_full[compartment]['inputs'], savedir+compartment+'/' + model + '/', model, data_time_to_mins=60,
                                            width=1.0, height=0.5, 
                                            data_downsample=10,
                                            ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
                                            y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                                            fname='_pred_traj_', labels=False, train_times=dat[compartment]['times']/data_time_to_mins)
                plt.close('all')


                post_df = pd.DataFrame({'Time (min)': dat_full[compartment]['times']/data_time_to_mins, 
                                    'Mean Response': np.mean(traj, axis=0), 
                                    '2.5th':np.percentile(traj, 2.5, axis=0), 
                                    '97.5th':np.percentile(traj, 97.5, axis=0)})
             
                post_df.to_csv(savedir+compartment+'/' + model + '/' + model +'_pred_traj.csv', index=False)
                
                # compute training error and testing error with RMSE and relative error
                # get index where training data ends
                train_end_idx = np.where(dat_full[compartment]['times'] == dat[compartment]['times'][-1])[0][0]
                dat[compartment]['times']

                # training error
                train_preds = traj[:,:train_end_idx+1]
                train_data = dat[compartment]['data']
                RMSE_train = np.sqrt(np.nanmean((np.nanmean(train_preds,axis=0) - train_data)**2))
                rel_err_train = np.linalg.norm(np.nanmean(train_preds,axis=0) - train_data)/np.linalg.norm(train_data)
                cred95_train = np.nanmean(np.squeeze(np.diff(np.nanquantile(train_preds, [0.025, 0.975], axis=0),axis=0)))
                std_train = np.nanmean(np.nanstd(train_preds, axis=0))
                
                # testing error
                test_preds = traj[:,train_end_idx+1:]
                test_data = dat_full[compartment]['data'][train_end_idx+1:]
                RMSE_test = np.sqrt(np.nanmean((np.nanmean(test_preds,axis=0) - test_data)**2))
                rel_err_test = np.linalg.norm(np.nanmean(test_preds,axis=0) - test_data)/np.linalg.norm(test_data)
                cred95_test = np.nanmean(np.squeeze(np.diff(np.nanquantile(test_preds, [0.025, 0.975], axis=0),axis=0)))
                std_test = np.nanmean(np.nanstd(test_preds, axis=0))

                # full data error
                RMSE_full_data = np.sqrt(np.nanmean((np.nanmean(traj,axis=0) - dat_full[compartment]['data'])**2))
                rel_err_full_data = np.linalg.norm(np.nanmean(traj,axis=0) - dat_full[compartment]['data'])/np.linalg.norm(dat_full[compartment]['data'])
                cred95_full_data = np.nanmean(np.squeeze(np.diff(np.nanquantile(traj, [0.025, 0.975], axis=0),axis=0)))
                std_full_data = np.nanmean(np.nanstd(traj, axis=0))

                # error on final 10 minutes
                idx_30_min = np.argmin(abs((dat_full[compartment]['times']/data_time_to_mins)-30))
                dat_final_10_min = dat_full[compartment]['data'][idx_30_min:]
                pred_final_10_min = traj[:,idx_30_min:]
                RMSE_final_10_min = np.sqrt(np.nanmean((np.nanmean(pred_final_10_min,axis=0) - dat_final_10_min)**2))
                rel_err_final_10_min = np.linalg.norm(np.nanmean(pred_final_10_min,axis=0) - dat_final_10_min)/np.linalg.norm(dat_final_10_min)
                cred95_final_10_min = np.nanmean(np.squeeze(np.diff(np.nanquantile(pred_final_10_min, [0.025, 0.975], axis=0),axis=0)))
                std_final_10_min = np.nanmean(np.nanstd(pred_final_10_min, axis=0))
                
                errors[compartment]['RMSE_train'][model] = RMSE_train
                errors[compartment]['rel_err_train'][model] = rel_err_train
                errors[compartment]['RMSE_test'][model] = RMSE_test
                errors[compartment]['rel_err_test'][model] = rel_err_test
                errors[compartment]['RMSE_full_data'][model] = RMSE_full_data
                errors[compartment]['rel_err_full_data'][model] = rel_err_full_data
                errors[compartment]['RMSE_final_10_min'][model] = RMSE_final_10_min
                errors[compartment]['rel_err_final_10_min'][model] = rel_err_final_10_min

                uncertainty[compartment]['cred95_train'][model] = cred95_train
                uncertainty[compartment]['std_train'][model] = std_train
                uncertainty[compartment]['cred95_test'][model] = cred95_test
                uncertainty[compartment]['std_test'][model] = std_test
                uncertainty[compartment]['cred95_full_data'][model] = cred95_full_data
                uncertainty[compartment]['std_full_data'][model] = std_full_data
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
