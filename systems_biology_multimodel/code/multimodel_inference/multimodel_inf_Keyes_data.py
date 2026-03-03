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

jax.config.update("jax_enable_x64", True)

plt.style.use('custom')

# custom plotting helper funcs
sys.path.insert(0, '../')
from plotting_helper_funcs import *
from utils import *

rng = np.random.default_rng(seed=1234)

model_info = json.load(open('../param_est/model_info.json', 'r'))
model_names = list(model_info.keys())

display_names = [model_info[model]['display_name'] for model in model_names]

# load in the full data
data = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
    'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
data['CYTO']['inputs'], data['CYTO']['data'], data['CYTO']['data_std'], \
    data['CYTO']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-CYTO.json', data_std=True, time=True)
data['PM']['inputs'], data['PM']['data'], data['PM']['data_std'], \
    data['PM']['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-PM.json', data_std=True, time=True)
data_time_to_mins = 60

# set up a color palette
# this is the ColorBrewer purple-green with 11 colors + three greys https://colorbrewer2.org/#type=diverging&scheme=PRGn&n=11
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#00441b','#363737','#929591','#d8dcd6']
# this one gets to 10 colors by removing the darkest purple
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#363737','#929591','#d8dcd6']

# create dictionary to store MMI weights
mmi_weights = {
    '40min':{'CYTO':{},'PM':{}},
    '30min':{'CYTO':{},'PM':{}},
    '20min':{'CYTO':{},'PM':{}},
    '10min':{'CYTO':{},'PM':{}}
}

##############################################
##### Compute standard MMI + make plots ######
##############################################
for time_len in ['', '_30min', '_20min', '_10min']:
    datadir = '../../../results/MAPK/param_est/Keyes_2020_data'+time_len+'/'
    savedir = '../../../results/MAPK/mmi/Keyes_2020_data'+time_len+'/'

    # load in the training data
    if time_len == '':
        data_train = data
    else:
        data_train = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
            'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
        for compartment in ['CYTO','PM']:
            data_train[compartment]['inputs'], data_train[compartment]['data'], data_train[compartment]['data_std'], \
                data_train[compartment]['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'+compartment+'-'+time_len.strip('_')+'.json', data_std=True, time=True)
        
        data_time_to_mins = 60

    # if savedir does not exist, create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    idata = {'CYTO':{},'PM':{}}
    posterior_samples = {'CYTO':{},'PM':{}}
    sample_times = {'CYTO':{},'PM':{}}
    ss = {'CYTO':{},'PM':{}}
    log_marginal_likes = {'CYTO':[],'PM':[]}

    for model in model_names:
        for compartment in ['CYTO','PM']:
            idata[compartment][model], ss[compartment][model], sample_times[compartment][model] = load_smc_samples_to_idata(datadir+compartment+'/' + model + '/' + model +'_smc_samples.json', sample_time=True)
            posterior_samples[compartment][model] = np.load(datadir+compartment+'/' + model + '/' + model +'_posterior_predictive_samples.npy')
            
            if model == 'hornberg_2005':
                # Hornberg 2005 only has 800 posterior predictive samples, so we duplicate and randomly permute them to get 2000
                H_2005_posts = [posterior_samples[compartment]['hornberg_2005'] for _ in range(3)]
                H_2005_post = np.vstack(H_2005_posts)
                posterior_samples[compartment]['hornberg_2005'] = H_2005_post[rng.permutation(np.arange(2000))]

            # now log marginal likelihoods
            _log_marg_like = ss[compartment][model]['log_marginal_likelihood']
            if len(_log_marg_like) == 1:
                log_marginal_likes[compartment].append(np.mean([chain[-1] for chain in _log_marg_like[0]]))
            else:
                log_marginal_likes[compartment].append(np.mean([chain[-1] for chain in _log_marg_like]))
        
    if time_len == '': # in this case some of the posterior predictive samples need to be modified because they are not the same length
        # shin has 4000 so downsample to 2000
        idxs = rng.choice(np.arange(4000), size=2000, replace=False)
        posterior_samples['CYTO']['shin_2014'] = posterior_samples['CYTO']['shin_2014'][idxs]
        posterior_samples['PM']['shin_2014'] = posterior_samples['PM']['shin_2014'][idxs]

    ##### Run arviz model comparison ######
    compare_data_waic_pbma = {}
    compare_data_loo_pbma = {}
    compare_data_waic_stack = {}
    compare_data_loo_stack = {}

    for compartment in ['CYTO','PM']:
        compare_data_waic_pbma[compartment] = az.compare(idata[compartment], ic='waic', method='BB-pseudo-BMA', seed=rng)
        compare_data_loo_pbma[compartment] = az.compare(idata[compartment], ic='loo', method='BB-pseudo-BMA', seed=rng)
        compare_data_waic_stack[compartment] = az.compare(idata[compartment], ic='waic', method='stacking', seed=rng)
        compare_data_loo_stack[compartment] = az.compare(idata[compartment], ic='loo', method='stacking', seed=rng)


    ##### Compute model probabilities ######
        # function to compute log sum exponential in a numerically stable way
    def logsumexp(x):
        c = np.max(x)
        return c + np.log(np.sum(np.exp(x - c)))

    def weight_with_logsumexp(log_values):
        return np.exp(log_values - logsumexp(log_values))
    
    m_probs = {}
    for compartment in ['CYTO','PM']:
        n_models = len(log_marginal_likes[compartment])
        prior_prob = 1/n_models
        model_probs = weight_with_logsumexp(np.log(prior_prob)+log_marginal_likes[compartment])
        m_probs[compartment] = model_probs
        

    ##### make plot of ELPPD, model probabilities, and model weights ######
    for compartment in ['CYTO','PM']: 
        # make a plot of elpd_loo, and elpd_waic
        dat = [compare_data_loo_stack[compartment].loc[model]['elpd_loo'] for model in model_names]
        fig, ax = get_sized_fig_ax(1.25, 0.9)
        bar = ax.bar(model_names, dat, edgecolor='k')
        ax.set_ylabel('expected \n log pointwise \n predictive density', fontsize=12)
        ax.set_xticklabels(display_names, rotation=90, fontsize=10.0)
        for i, br in enumerate(bar):
            br.set_facecolor(colors[i])
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], '--k', linewidth=0.5)
        ax.set_xlim(xlim)
        # ylim = ax.get_ylim()
        # ax.set_ylim(-0.04*ylim[1], 0.5*ylim[1])
        # ax.set_yticks([0, 200.0])
        # ax.set_yticklabels([0, 200.0], fontsize=10.0)
        fig.savefig(savedir + compartment + '_traj_elpd_loo.pdf', transparent=True)

        # waic
        dat = [compare_data_waic_stack[compartment].loc[model]['elpd_waic'] for model in model_names]
        fig, ax = get_sized_fig_ax(1.25, 0.9)
        bar = ax.bar(model_names, dat, edgecolor='k')
        for i, br in enumerate(bar):
            br.set_facecolor(colors[i])

        ax.set_ylabel('expected \n log pointwise \n predictive density \n (WAIC)', fontsize=12)
        ax.set_xticklabels(display_names, rotation=90, fontsize=10.0)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], '--k', linewidth=0.5)
        ax.set_xlim(xlim)
        # ylim = ax.get_ylim()
        # ax.set_ylim(-0.01*ylim[1], 0.15*ylim[1])
        # ax.set_yticks([0, 200.0])
        # ax.set_yticklabels([0, 200.0], fontsize=10.0)
        fig.savefig(savedir + compartment + '_traj_elpd_waic.pdf', transparent=True)

        # model probabilities
        fig, ax = get_sized_fig_ax(1.25, 0.9)
        bar = ax.bar(model_names, model_probs, edgecolor='k')
        for i, br in enumerate(bar):
            br.set_facecolor(colors[i])

        ax.set_ylabel('model \n probability', fontsize=12)
        xlim = ax.get_xlim()
        ax.plot(xlim, [0,0], '--k', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim([0, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels([0, 0.5, 1.0], fontsize=10.0)
        ax.set_xticklabels(display_names, rotation=90, fontsize=10.0)
        # ax.set_ylim([0, 1.0])
        fig.savefig(savedir + compartment + '_traj_model_probs.pdf', transparent=True)

        # model weights
        x = np.arange(n_models)  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        stack = [compare_data_loo_stack[compartment].loc[model]['weight'] for model in model_names]
        pBMA = [compare_data_loo_pbma[compartment].loc[model]['weight'] for model in model_names]

        # 
        fig, ax = get_sized_fig_ax(2.5, 0.9)
        for dat, name, col in zip([pBMA, stack, m_probs[compartment]], ['pseudo-BMA','stacking','BMA',], [colors[-3], colors[-2], colors[-1]]):
            offset = width * multiplier
            bar = ax.bar(x + offset, dat, width, label=name, color=col, edgecolor='k')
            # ax.bar_label(bar, padding=3)
            multiplier += 1

        ax.set_ylabel('model weight', fontsize=12)
        ax.set_xticks(x + width, display_names)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=10.0)
        leg = ax.legend(loc='center', fontsize=8.0, bbox_to_anchor=(0.5, 1.3), ncol=1)
        leg.remove()

        ax.set_ylabel(r'weight', fontsize=12)
        ax.set_xticklabels(display_names, rotation=90, fontsize=10.0)
        # ax.set_ylim([0, 1.0])
        fig.savefig(savedir + compartment + '_traj_model_weights.pdf', transparent=True)

    ##### plot posterior predictive MMI traces ######
    # first with posterior predictive samples
    loo_bma_combined = {}
    loo_stack_combined = {}
    loo_pbma_combined = {}
    trajs_mmi = {'CYTO':{},'PM':{}}
    for compartment in ['CYTO','PM']:
        loo_pbma_combined[compartment] = np.zeros_like(posterior_samples[compartment]['kholodenko_2000'])
        loo_stack_combined[compartment] = np.zeros_like(posterior_samples[compartment]['kholodenko_2000'])
        loo_bma_combined[compartment] = np.zeros_like(posterior_samples[compartment]['kholodenko_2000'])
        for i, name in enumerate(model_names):
            print(name, time_len)
            loo_pbma_combined[compartment] += compare_data_loo_pbma[compartment].loc[name]['weight']*posterior_samples[compartment][name]
            loo_stack_combined[compartment] += compare_data_loo_stack[compartment].loc[name]['weight']*posterior_samples[compartment][name]
            loo_bma_combined[compartment] += m_probs[compartment][i]*posterior_samples[compartment][name]

        print(loo_pbma_combined[compartment].shape)
        trajs_mmi[compartment] = {
            'pseudo-BMA': loo_pbma_combined[compartment],
            'stacking': loo_stack_combined[compartment],
            'BMA': loo_bma_combined[compartment]
        }

        cols = [colors[-3], colors[-2], colors[-1]]
        for idx, name in enumerate(trajs_mmi[compartment].keys()):
            plot_posterior_trajectories(trajs_mmi[compartment][name], data_train[compartment]['data'], data_train[compartment]['data_std'], 
                                        data_train[compartment]['times'], cols[idx], data_train[compartment]['inputs'], 
                                        savedir + compartment, name, data_time_to_mins=60,
                                                    width=1.1, height=0.5, 
                                                    data_downsample=10,
                                                    ylim=[[0.0, 1.5]],
                                                    y_ticks=[[0.0, 1.0]],
                                                    fname='_mmi_traj_', 
                                                    labels=False, xlim=[0,40])
    
    ##### Bar plots with errors and uncertainties ######
    #TODO: add me
    
    ##### Store MMI weights ######
    if time_len == '':
        for compartment in ['CYTO','PM']:
            mmi_weights['40min'][compartment] = {'m_probs':m_probs[compartment],
                                                    'loo_pbma_combined':loo_pbma_combined[compartment],
                                                    'loo_stack_combined':loo_stack_combined[compartment],
                                                    'loo_bma_combined':loo_bma_combined[compartment]}

##############################################
######## Analyze train/test settings #########
##############################################
for time_len in ['', '_30min', '_20min', '_10min']:
    datadir = '../../../results/MAPK/param_est/Keyes_2020_data'+time_len+'/'
    savedir = '../../../results/MAPK/mmi/Keyes_2020_data'+time_len+'/'

    # load in the training data
    if time_len == '':
        data_train = data
    else:
        data_train = {'CYTO':{'inputs':None, 'data':None,'data_std':None, 'times':None},
            'PM':{'inputs':None, 'data':None,'data_std':None, 'times':None}}
        for compartment in ['CYTO','PM']:
            data_train[compartment]['inputs'], data_train[compartment]['data'], data_train[compartment]['data_std'], \
                data_train[compartment]['times'] = load_data_json('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'+compartment+'-'+time_len.strip('_')+'.json', data_std=True, time=True)
        
        data_time_to_mins = 60

    idata = {'CYTO':{},'PM':{}}
    posterior_samples = {'CYTO':{},'PM':{}}
    sample_times = {'CYTO':{},'PM':{}}
    ss = {'CYTO':{},'PM':{}}
    log_marginal_likes = {'CYTO':[],'PM':[]}

    for model in model_names:
        for compartment in ['CYTO','PM']:
            idata[compartment][model], ss[compartment][model], sample_times[compartment][model] = load_smc_samples_to_idata(datadir+compartment+'/' + model + '/' + model +'_smc_samples.json', sample_time=True)
            posterior_samples[compartment][model] = np.load(datadir+compartment+'/' + model + '/' + model +'_posterior_predictive_samples.npy')
            
            if model == 'hornberg_2005':
                # Hornberg 2005 only has 800 posterior predictive samples, so we duplicate and randomly permute them to get 2000
                H_2005_posts = [posterior_samples[compartment]['hornberg_2005'] for _ in range(3)]
                H_2005_post = np.vstack(H_2005_posts)
                posterior_samples[compartment]['hornberg_2005'] = H_2005_post[rng.permutation(np.arange(2000))]

            # now log marginal likelihoods
            _log_marg_like = ss[compartment][model]['log_marginal_likelihood']
            if len(_log_marg_like) == 1:
                log_marginal_likes[compartment].append(np.mean([chain[-1] for chain in _log_marg_like[0]]))
            else:
                log_marginal_likes[compartment].append(np.mean([chain[-1] for chain in _log_marg_like]))