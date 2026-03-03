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

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")

# # print out device count
# n_devices = jax.local_device_count() 
# print(jax.devices())
# print('Using {} jax devices'.format(n_devices))

##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Generate Morris samples for the specified model.")
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-data_file", type=str, help="path to the data file. Should be a CSV with one column of inputs and another of outputs data.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of samples to posterior samples to draw. Defaults to 1000.")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    parser.add_argument("-input_state", type=str, default='EGF', help="Name of EGF input in the state vector. Defaults to EGF.")
    parser.add_argument("-EGF_conversion_factor", type=float, default=1.0, help="Conversion factor to convert EGF from nM to other units. Defaults to 1.")
    parser.add_argument("-ERK_states", type=str, default=None, help="Names of ERK species to use for inference. Defaults to None.")
    parser.add_argument("-time_conversion_factor", type=float, default=1.0, help="Conversion factor to convert from seconds by division. Default is 1. Mins would be 60")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to uniform.")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores.")
    parser.add_argument("-nchains", type=int, default=4, help="Number of chains to run. Defaults to 4.")
    parser.add_argument("--skip_prior_sample", action='store_false',default=True) 
    parser.add_argument("--skip_sample", action='store_false',default=True)
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-upper_prior_mult", type=float,default=1e2)
    parser.add_argument("-lower_prior_mult", type=float,default=1e-2)
    parser.add_argument('-pcoeff', type=float, default=0, help='pcoeff for PID time stepper')
    parser.add_argument('-dcoeff', type=float, default=0, help='dcoeff for PID time stepper')
    parser.add_argument('-icoeff', type=float, default=1.0, help='icoeff for PID time stepper')
    parser.add_argument('-smc_threshold', type=float, default=0.85, help='Threshold for SMC sampling')

    args=parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    """ main function to execute command line script functionality.
    """
    args = parse_args(raw_args)

    print('Processing model {}.'.format(args.model))
    
    # try calling the model
    try:
        model = eval(args.model + '(transient=False)')
    except:
        print('Warning Model {} not found. Skipping this.'.format(args.model))

    # get parameter names and initial conditions
    p_dict, plist = model.get_nominal_params()
    y0_dict, y0 = model.get_initial_conditions()

    # add savedir if it does not exist
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    # load the data
    # HF96 model is in seconds, so times is in seconds, so we set the data_time_to_mins to 60
    inputs, data, data_std, times = load_data_json(args.data_file, data_std=True, time=True)
    data_time_to_mins = 60

    # convert EGF to required units
    inputs_native_units = inputs * args.EGF_conversion_factor

    # get the params to sample
    analyze_params = args.free_params.split(',')
    free_param_idxs = [list(p_dict.keys()).index(p) for p in analyze_params]

    # get the EGF index and ERK indices
    state_names = list(y0_dict.keys())
    EGF_idx = state_names.index(args.input_state)
    ERK_indices = [state_names.index(s) for s in args.ERK_states.split(',')]

    # construct the strings to make priors and constants
    prior_param_dict = set_prior_params(args.model, list(p_dict.keys()), plist, free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family, savedir=args.savedir, saveplot=False)
    
    # make simulator lambda function that solves at correct times with the time conversion factor taken into account
    if len(inputs) > 1:
        ERK_stim_traj = lambda p,model, max_time, y0, output_states: ERK_stim_trajectory_set(p, model, max_time, y0, output_states, times/args.time_conversion_factor, max_input_index=max_input_idx, rtol=args.rtol, atol=args.atol)
    else:
        print('Using single input traj func.')
        def ERK_stim_traj(p, model, max_time, y0, output_states):
            traj = solve_traj(model, y0, p, max_time, output_states, times/args.time_conversion_factor, args.rtol, args.atol, pcoeff=args.pcoeff, dcoeff=args.dcoeff, icoeff=args.icoeff)
            # return normalized trajectory
            return [(traj - np.min(traj)) / (np.max(traj) - np.min(traj))], traj

    
    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)
    max_input_idx = np.argmax(inputs_native_units) # get index of max input

    # construct the pymc model
    # Note: We do not use the build_pymc_model function, because we need to 
    #   build a model that runs the simulator three times for each input level
    if len(inputs) > 1:
        y0_use = y0_EGF_ins
    else:
        y0_use = y0_EGF_ins[0]
    pymc_model = build_pymc_model(prior_param_dict, [data], y0_use, 
                    ERK_indices, np.max(times/args.time_conversion_factor), diffrax.ODETerm(model), 
                    simulator=ERK_stim_traj, data_sigma=[data_std])
    
    if args.skip_prior_sample:
        create_prior_predictive(pymc_model, args.model, data, inputs, args.savedir, 
            trajectory=True, times=times/data_time_to_mins, data_std=[data_std], nsamples=200)

    # SMC sampling
    if args.skip_sample:
        posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                    nsamples=args.nsamples, ncores=args.ncores, threshold=args.smc_threshold, chains=args.nchains,)
    else:
        posterior_idata, _ = load_smc_samples_to_idata(args.savedir + args.model + '_smc_samples.json')
    
    # trace plots and diagnostics
    plot_sampling_trace_diagnoses(posterior_idata, args.savedir, args.model)

    print(times)
    # posterior predictive samples
    create_posterior_predictive(pymc_model, posterior_idata, args.model, data, inputs, args.savedir, 
            trajectory=True, times=times/data_time_to_mins, data_std=data_std)
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()