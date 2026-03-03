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
import csv

rng = np.random.default_rng(seed=1234)

################ LOAD in DATA ################
savedir = '../../../results/MAPK/param_est/HF96_synthetic_data/'

# load in the model info 
model_info = json.load(open('model_info.json', 'r'))
model_names = list(model_info.keys())
model_names.remove('huang_ferrell_1996')
display_names = [model_info[model]['display_name'] for model in model_names]
print(model_names)

idata = {}
posterior_samples = {}
sample_times = {}

for model in model_names:
    print(model)
    idata[model], _, sample_times[model] = load_smc_samples_to_idata(savedir+model+'/'+model+'_smc_samples.json', sample_time=True)
    posterior_samples[model] = np.load(savedir+model+'/'+model+'_posterior_predictive_samples.npy')

# shin has 16000 so downsample to 4000
idxs = rng.choice(np.arange(16000), size=4000, replace=False)
posterior_samples['shin_2014'] = posterior_samples['shin_2014'][idxs]

# get training and testing data
inputs, data = load_data('../../../results/MAPK/HF_96_synthetic_data.csv')
inputs_traj, data_traj, data_std_traj, times_traj = load_data_json('../../../results/MAPK/HF_96_traj_data.json', data_std=True, time=True)
data_time_to_mins = 60

# set up a color palette
# this is the ColorBrewer purple-green with 11 colors + three greys https://colorbrewer2.org/#type=diverging&scheme=PRGn&n=11
# this one gets to 10 colors by removing the darkest green
colors = ['#40004b','#762a83','#9970ab','#c2a5cf','#e7d4e8','#f7f7f7','#d9f0d3','#a6dba0','#5aae61','#1b7837','#363737','#929591','#d8dcd6']

# get the standard color pallette
colors = get_color_pallette()

################ Write sampling times to a file ################
with open(savedir + 'SMC_runtimes.txt', 'w') as f:
    for model in model_names:
        f.write(f'{model}: {sample_times[model]/3600} hr\n')

plotting_params = {
    'kholodenko_2000':[False,False,False,False],'levchenko_2000':[False,False,False,False],'hatakeyama_2003':[False,False,False,False],'hornberg_2005':[False,False,False,False],'birtwistle_2007':[False,False,False,False],'orton_2009':[False,False,False,False],'vonKriegsheim_2009':[False,False,False,False],'shin_2014':[False,False,False,False],'ryu_2015':[False,False,False,False],'kochanczyk_2017':[False,False,False,False]
}

# ################ Make pretty posterior predictive dose-response curves ################
# skip_idxs = []
# for idx,model in enumerate(model_names):
#     if idx in skip_idxs:
#         print('skipping', model)
#         continue
#     else:
#         print('plotting', model)

#         plot_p = plotting_params[model]

#         fig, ax = plot_stimulus_response_curve(posterior_samples[model], data, inputs, input_name='EGF stimulus (nM)', output_name='% maximal ERK \n activity', box_color='w', data_color='#5aae61',
#                                         data_std=0.1, width=1.1, height=1.1, data_marker_size=5.0, scatter_marker_size=0,
#                                         title=None, xlabel=plot_p[0],xticklabels=plot_p[1],ylabel=plot_p[2], yticklabels=plot_p[3])
#         ax.set_title(ax.get_title(), fontsize=12.0)
#         fig.savefig(savedir+model+'/'+model+'_posterior_predictive.pdf', transparent=True)

# plt.close('all')

########################### Make plot of the data ##############################
# use the pretty posterior predictive dose-response function
fig, ax = plot_stimulus_response_curve(-100*np.ones_like(posterior_samples['shin_2014']), data, inputs, input_name='EGF stimulus (nM)', output_name='% maximal ERK \n activity', box_color='w', data_color='#5aae61',
                                        data_std=0.1, width=1.1, height=1.1, data_marker_size=5.0, scatter_marker_size=0,
                                        title=None, xlabel=False,xticklabels=False,ylabel=False, yticklabels=False)
ax.set_title(ax.get_title(), fontsize=12.0)

fig.savefig(savedir+'dose_response_training_data.pdf', transparent=True)

# make a legend
fig, ax = plt.subplots()
ax.errorbar(-100, -100, yerr=0.1, color='red', fmt='x', markersize=5.0, label='training data')
leg = ax.legend(bbox_to_anchor=(2.0, 1.0), numpoints=1, fontsize=8.0)
export_legend(leg, savedir+'dose_response_training_data_leg.pdf')
leg.remove()
plt.close('fig')
 

################ Make pretty posterior dose-response curves ################
## We also need to plot and analyze dose-responses that are not posterior predictive, 
#           but simply use posterior samples to compute them

skip_idxs = []
for idx,model in enumerate(model_names):
    if idx in skip_idxs:
        print('skipping', model)
        continue
    else:
        print('plotting', model)
        this_model_info = model_info[model]

        plot_p = plotting_params[model]

        fname = savedir+model+'/dose_response_predict.npy'
        if os.path.exists(fname):
            dose_response = np.load(fname)
        else:
            max_time = this_model_info['max_time']
            if max_time == 'jnp.inf':
                max_time = np.inf
            else:
                max_time = float(max_time)

            ss_method = this_model_info['ss_method']
            if ss_method == 'newton':
                event_rtol = float(this_model_info['newton_event_rtol'])
                event_atol = float(this_model_info['newton_event_atol'])
            else:
                event_rtol = float(this_model_info['event_rtol'])
                event_atol = float(this_model_info['event_atol'])

            # create dose-response curve prediction
            dose_response = predict_dose_response(model, idata[model], inputs,   
                                    this_model_info['input_state'], this_model_info['ERK_states'], 
                                    max_time, EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']),nsamples=400, timeout=30,
                                    ss_method=ss_method,event_atol=event_atol, event_rtol=event_rtol,)
            # # save
            np.save(fname, dose_response)

        fig, ax = plot_stimulus_response_curve(dose_response, data, 
                                               inputs, input_name='EGF stimulus (nM)', 
                                               output_name='% maximal ERK \n activity', 
                                               box_color='w', data_color='#5aae61',
                                        data_std=0.1, width=1.1, height=1.1, 
                                        data_marker_size=5.0, scatter_marker_size=0,
                                        title=None, xlabel=plot_p[0],xticklabels=plot_p[1],ylabel=plot_p[2], yticklabels=plot_p[3])
        ax.set_title(ax.get_title(), fontsize=12.0)
        fig.savefig(savedir+model+'/'+model+'_posterior.pdf', transparent=True)

        fig, ax = plot_stimulus_response_curve(dose_response, data, inputs, input_name='EGF stimulus (nM)', output_name='% maximal ERK \n activity', box_color='w', data_color='red',
                                        data_std=0.1, width=1.1, height=1.1, data_marker_size=5.0, scatter_marker_size=0,
                                        title=None, xlabel=plot_p[0],xticklabels=plot_p[1],ylabel=plot_p[2], yticklabels=plot_p[3])
        ax.set_title(ax.get_title(), fontsize=12.0)
        fig.savefig(savedir+model+'/'+model+'_dose_response_predict.pdf', transparent=True)

        # create csv of dose-response predictions for source_data.xlsx
        post_df = pd.DataFrame({'EGF (nM)': inputs, 'Mean Response': np.mean(dose_response, axis=0), '2.5th':np.percentile(dose_response, 2.5, axis=0), '97.5th':np.percentile(dose_response, 97.5, axis=0)})
        post_df.to_csv(savedir+model+'/'+model+'_dose_response_predict.csv', index=False)


# ################ Make posterior trajectory ################
# ## Now we want to use posterior draws to simulate trajectory predictions
# n_traj = 400

# skip_idxs = []
# for idx,model in enumerate(model_names):
#     if idx in skip_idxs:
#         print('skipping', model)
#         continue
#     else:
#         print('plotting', model)
#         this_model_info = model_info[model]

#         plot_p = plotting_params[model]

     
#         # predict trajectories
#         fname = savedir+model+'/traj_predict.npy'
#         if os.path.exists(fname):
#             traj = np.load(fname)
#         else:
#             traj = predict_traj_response(model, idata[model], inputs_traj, times_traj, 
#                                                 this_model_info['input_state'], this_model_info['ERK_states'],
#                                                 float(this_model_info['time_conversion']),
#                                                 EGF_conversion_factor=float(this_model_info['EGF_conversion_factor']),
#                                                 nsamples=400)
#             # save
#             np.save(savedir+model+'/traj_predict.npy', traj)

#         # plot
#         plot_posterior_trajectories(traj, data_traj, data_std_traj, times_traj, colors[idx], 
#                                         inputs_traj, savedir+model+'/',
#                                         model, data_time_to_mins=60,
#                                         width=1.1, height=0.5, 
#                                         data_downsample=10,
#                                         ylim=[[0.0, 1.2], [0.0, 1.2], [0.0, 1.2]],
#                                         y_ticks=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
#                                         fname='_pred_traj_', labels=False)
# plt.close('all')