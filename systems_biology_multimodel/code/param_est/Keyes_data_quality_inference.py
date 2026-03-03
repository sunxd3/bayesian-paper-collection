import pdb
from os import environ
environ['OMP_NUM_THREADS'] = '1'
#environ['CUDA_VISIBLE_DEVICES'] = '0'

import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import sys
import argparse
from tqdm import tqdm


sys.path.append("../models/")
from huang_ferrell_1996 import *
from bhalla_iyengar_1999 import *
from kholodenko_2000 import *
from levchenko_2000 import *
from brightman_fell_2000 import *
from schoeberl_2002 import *
from hatakeyama_2003 import *
from hornberg_2005 import *
from birtwistle_2007 import *
from orton_2009 import *
from vonKriegsheim_2009 import *
from shin_2014 import *
from ryu_2015 import *
from kochanczyk_2017 import *
from dessauges_2022 import *

sys.path.append("../")
from utils import *
import os
import os
import shutil
import json

import inference_process_traj as inference_process

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")   


##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Run data subsampling study.")
    parser.add_argument("-compartment", type=str, help="compartment to process. (CYTO or PM)")
    parser.add_argument("-savedir", type=str, help="Path to save results.")
    parser.add_argument("-n_sets", type=int, default=25, help="Number of sets to sample. Defaults to 25.")
    parser.add_argument("-subset_sizes", type=str, default="[10,20,40,60]", help="List of subset sizes to sample. Defaults to [10,20,40,60].")
    parser.add_argument("-model_names", type=str, default='kholodenko_2000,orton_2009,shin_2014,ryu_2015,kochanczyk_2017', help="List of model names to process. Defaults to  'kholodenko_2000,levchenko_2000,orton_2009,shin_2014,ryu_2015,kochanczyk_2017'.")
    parser.add_argument("-EGF_input", type=float, default=1.653e4, help="EGF input level. Defaults to 1.653e4 nM EGF.")
    parser.add_argument("--reprocess_run", action='store_false',default=True, help="Flag to indicate if this is a new run. IF it is a new run then past run info is moved to past_run subdirs and inference is run again. Any data in past_run subdirs is removed. If it is not a new run, then inference is not run again and the past run info is NOT moved to past_run directory.")
    
    args=parser.parse_args(raw_args)
    return args

def check_and_move_directory_contents(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    # Check if the directory is empty
    if not os.listdir(directory):
        print(f"Directory '{directory}' is empty.")
        return
    
    # If directory is not empty, create subdirectory 'past_run' if not exists
    past_run_dir = os.path.join(directory, 'past_run')
    if not os.path.exists(past_run_dir):
        os.makedirs(past_run_dir)
    
    # Move contents of directory to 'past_run'
    for item in os.listdir(directory):
        if item == 'past_run':
            continue
        source = os.path.join(directory, item)
        destination = os.path.join(past_run_dir, item)
        if os.path.isdir(source):
            shutil.move(source, destination)
            print(f"Moved directory '{item}' to 'past_run'.")
        else:
            shutil.move(source, destination)
            print(f"Moved file '{item}' to 'past_run'.")


def main(raw_args=None):
    """ main function to execute command line script functionality.
    """
    args = parse_args(raw_args)

    print('Processing model {}.'.format(args.compartment))

    # savedir
    savedir = args.savedir + args.compartment + '/'
    # add savedir if it does not exist
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    # load the data
    if args.compartment == 'CYTO':
        data = np.load('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-CYTO_normalized.npz')
    elif args.compartment == 'PM':
        data = np.load('../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-PM_normalized.npz')
        
    # preprocess the data
    times = data['time']
    responses = data['data']

    # get info about the data
    ntimes, ncells = responses.shape

    # convert from str to list
    subset_sizes = eval(args.subset_sizes)

    # load model info
    with open('./model_info.json') as file:
        model_info = json.load(file)
    models = args.model_names.split(',')
    
    keys = models + ['pseudoBMA', 'BMA', 'stacking']
    # dictionaries to hold data
    # relative error to mean of data with all cells
    rel_error_full_dat = {size:{key:[] for key in keys} for size in subset_sizes}
    print(rel_error_full_dat)
    # relative error to mean of data with subset of cells
    rel_error_sub_dat = {size:{key:[] for key in keys} for size in subset_sizes}
    # RMSE
    RMSE_full_dat = {size:{key:[] for key in keys} for size in subset_sizes} 
    RMSE_sub_dat = {size:{key:[] for key in keys} for size in subset_sizes}
    # 95% cred int width
    uncert95 = {size:{key:[] for key in keys} for size in subset_sizes}
    # std
    std = {size:{key:[] for key in keys} for size in subset_sizes}

    # summary stats about the data subsets
    data_RMSE = {size:[] for size in subset_sizes}
    data_rel_error = {size:[] for size in subset_sizes}
    data_uncert95 = {size:[] for size in subset_sizes}
    data_std = {size:[] for size in subset_sizes}


    # this is the main loop over the subset sizes
    for size in tqdm(subset_sizes):
        print('Processing subset size {}'.format(size))

        # create a dir for all results are this subset size
        _savedir = savedir + 'subset_size_{}/'.format(size)

        if args.reprocess_run:
            # add _savedir if it does not exist
            if not os.path.isdir(_savedir):
                os.makedirs(_savedir)
            else:
                if not os.path.isdir(_savedir + 'past_run/'):
                    check_and_move_directory_contents(_savedir)
                else:
                    # delete directory _savedir+'past_run/'
                    shutil.rmtree(_savedir + 'past_run/')
                    check_and_move_directory_contents(_savedir)
        
        # this is the inner loop over the number of sets st the given subset size
        for i in range(args.n_sets):
            if args.reprocess_run:
                # daw a random sample of cells
                idxs = rng.choice(ncells, int(size), replace=False,)
                dat = responses[:,idxs]

                # save the data to _savedir
                data = {'stimulus': [args.EGF_input], 'response': list(np.nanmean(dat, axis=1)), 'time': list(times*60), 'response_std': list(np.nanstd(dat, axis=1))}
                with open(_savedir + 'sample_{}.json'.format(i),'w') as data_file:
                    json.dump(data, data_file)

                np.save(_savedir + 'sample_{}.npy'.format(i), dat)
            else:
                # load the data
                dat = np.load(_savedir + 'sample_{}.npy'.format(i))
                print(dat.shape)

            # compute the summary stats of the data subset
            data_RMSE[size].append(np.sqrt(np.nanmean((np.nanmean(dat,axis=1) \
                            - np.nanmean(responses,axis=1))**2)))
            data_rel_error[size].append(np.linalg.norm(np.nanmean(dat,axis=1) \
                            - np.nanmean(responses,axis=1))/np.linalg.norm(np.nanmean(responses,axis=1)))
            data_uncert95[size].append(np.nanmean(np.abs(np.squeeze(np.diff(np.nanquantile(dat, [0.025, 0.975], axis=0),axis=1)))))
            data_std[size].append(np.nanmean(np.nanstd(dat, axis=1)))

            # inference loop -- this is where we run inference on the data for each model
            #    we will call the inference_process_HF_synth_traj.py script
            for model in models:

                if args.reprocess_run:
                    # run inference
                    m_info = model_info[model]

                    arg_lst = ['-model', model,  
                    '-free_params', m_info['free_params'], 
                    '-data_file', _savedir + 'sample_{}.json'.format(i), 
                    '-nsamples', str(500), 
                    '-ncores', str(4), 
                    '-savedir', _savedir + 'sample_{}_'.format(i), 
                    '-input_state', m_info['input_state'],
                    '-ERK_states', m_info['ERK_states'],
                    '-prior_family', m_info['prior_family'], 
                    #  '-max_time', m_info['max_time'],
                    '-EGF_conversion_factor', m_info['EGF_conversion_factor'],
                    '-time_conversion_factor', m_info['time_conversion'],
                    '--skip_prior_sample']
                    
                    # run the inference
                    inference_process.main(arg_lst)

                # load the posterior samples
                # idata, _  = load_smc_samples_to_idata(_savedir + 'sample_{}_'.format(i) + model + '_smc_samples.json', sample_time=False)

                # load posterior predictive samples
                post_samples = np.load(_savedir + 'sample_{}_'.format(i) + model + '_posterior_predictive_samples.npy')

                # compute the relative error, RMSE and uncert95 of individual models
                RMSE_full_dat[size][model].append(np.sqrt(np.nanmean((np.nanmean(post_samples,axis=0) \
                                - np.nanmean(responses,axis=1))**2)))
                RMSE_sub_dat[size][model].append(np.sqrt(np.nanmean((np.nanmean(post_samples,axis=0) \
                                - np.nanmean(dat,axis=1))**2)))
                rel_error_full_dat[size][model].append(np.linalg.norm(np.nanmean(post_samples,axis=0) \
                                - np.nanmean(responses,axis=1))/np.linalg.norm(np.nanmean(responses,axis=1)))
                rel_error_sub_dat[size][model].append(np.linalg.norm(np.nanmean(post_samples,axis=0) \
                                - np.nanmean(dat,axis=1))/np.linalg.norm(np.nanmean(dat,axis=1)))
                uncert95[size][model].append(np.nanmean(np.abs(np.squeeze(np.diff(np.nanquantile(post_samples, [0.025, 0.975], axis=0),axis=0)))))
                std[size][model].append(np.nanmean(np.nanstd(post_samples, axis=0)))
            
    # save errors and uncertainties to file
    # data summary stats
    with open(savedir + 'data_RMSE.json', 'w') as f:
        json.dump(data_RMSE, f)
    with open(savedir + 'data_rel_error.json', 'w') as f:
        json.dump(data_rel_error, f)
    with open(savedir + 'data_uncert95.json', 'w') as f:
        json.dump(data_uncert95, f)
    with open(savedir + 'data_std.json', 'w') as f:
        json.dump(data_std, f)

    # model predictions
    with open(savedir + 'RMSE_full_dat.json', 'w') as f:
        json.dump(RMSE_full_dat, f)
    with open(savedir + 'RMSE_sub_dat.json', 'w') as f:
        json.dump(RMSE_sub_dat, f)
    with open(savedir + 'rel_error_full_dat.json', 'w') as f:
        json.dump(rel_error_full_dat, f)
    with open(savedir + 'rel_error_sub_dat.json', 'w') as f:
        json.dump(rel_error_sub_dat, f)
    with open(savedir + 'uncert95.json', 'w') as f:
        json.dump(uncert95, f)
    with open(savedir + 'std.json', 'w') as f:
        json.dump(std, f)
    
    
    print('Completed {}'.format(args.compartment))

if __name__ == '__main__':
    main()
