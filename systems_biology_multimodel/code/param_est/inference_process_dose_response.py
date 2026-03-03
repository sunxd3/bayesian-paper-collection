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

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "cpu")

# print out device count
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
    parser.add_argument("-t1", type=int, default=jnp.inf, help="Time to simulate the model. Defaults to None.")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to uniform.")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores.")
    parser.add_argument("--skip_prior_sample", action='store_false',default=True) 
    parser.add_argument("--skip_ss_check_func", action='store_false',default=True) 
    parser.add_argument("--skip_sample", action='store_false',default=True)
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-ss_method", type=str, default='ode')
    parser.add_argument("-event_rtol", type=float,default=1e-6)
    parser.add_argument("-event_atol", type=float,default=1e-5)
    parser.add_argument("-newton_event_rtol", type=float,default=1e-5)
    parser.add_argument("-newton_event_atol", type=float,default=1e-5)
    parser.add_argument("-ss_check_sim_tmax", type=float,default=1e6)
    parser.add_argument("-ss_check_sim_samples", type=int,default=10)
    parser.add_argument("-ss_check_sim_yaxis_scale", type=str, default=None)

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

    # load the data
    inputs, data = load_data(args.data_file)

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
    prior_param_dict = set_prior_params(args.model, list(p_dict.keys()), plist, free_param_idxs,
                                        upper_mult=100, lower_mult=0.01,prior_family=args.prior_family, saveplot=False,
                                        savedir=args.savedir)

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)

    # simulator function
    if args.ss_method == 'ode':
        event_rtol = args.event_rtol
        event_atol = args.event_atol
    elif args.ss_method == 'newton':
        event_rtol = args.newton_event_rtol
        event_atol = args.newton_event_atol
    
    print('Using {} for SS with tols atol={}, rtol={}.'.format(args.ss_method, event_atol, event_rtol))
    simulator = lambda params, model_dfrx_ode, max_time, y0_EGF_inputs, \
        output_states: ERK_stim_response(params, model_dfrx_ode, max_time, y0_EGF_inputs, 
                      output_states, event_rtol=event_rtol, event_atol=event_atol,
                      rtol=args.rtol, atol=args.atol, ode_or_newton=args.ss_method)
    
    # construct the pymc model
    pymc_model = build_pymc_model(prior_param_dict, data, y0_EGF_ins, 
                    ERK_indices, args.t1, diffrax.ODETerm(model), simulator=simulator)
    if args.skip_ss_check_func:
        prior_check_ss_func(args.model, diffrax.ODETerm(model), pymc_model, jnp.array(plist), 
                            y0_EGF_ins[4], args.t1, free_param_idxs, ERK_indices, args.savedir, 
                            print_results=True, nsamples=args.ss_check_sim_samples, tmax=args.ss_check_sim_tmax,
                            event_rtol=args.event_rtol, event_atol=args.event_atol,
                            newton_rtol=args.newton_event_rtol, newton_atol=args.newton_event_atol,
                            yaxis_scale=args.ss_check_sim_yaxis_scale, rtol=args.rtol, atol=args.atol)
    
    # # prior predictive sampling
    if args.skip_prior_sample:
        create_prior_predictive(pymc_model, args.model, data, inputs, args.savedir, 
                                nsamples=500)
    
    # SMC sampling
    if args.skip_sample:
        posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                    nsamples=args.nsamples, ncores=args.ncores, threshold=0.85, chains=4,)
    
        # trace plots and diagnostics
        plot_sampling_trace_diagnoses(posterior_idata, args.savedir, args.model)
    
    # load posterior idata if it is not here from sampling
    if 'posterior_idata' not in locals():
        try:
            posterior_idata, _ = load_smc_samples_to_idata(args.savedir + args.model + '_smc_samples.json', sample_time=False)
        except:
            print('Warning: posterior idata not found. Skipping this.')
            return
    
    # posterior predictive sampling
    create_posterior_predictive(pymc_model, posterior_idata, args.model, data, 
                                inputs, args.savedir)

    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    main()

