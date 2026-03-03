import pdb
from os import environ
environ['OMP_NUM_THREADS'] = '1'
#environ['CUDA_VISIBLE_DEVICES'] = '0'
import multiprocessing
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
import sys
import argparse
import pymc as pm
from pytensor.tensor import max as pt_max

sys.path.append("../models/")
from orton_2009 import *
# Rap1 models
from shin_2014_Rap1 import *
from ryu_2015_Rap1 import *
from vonKriegsheim_2009_Rap1 import *
# from kochanczyk_2017_rap1 import *

sys.path.append("../")
from utils import *
import os

# tell jax to use 64bit floats
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# need a posterior predictive function that can deal with the fact that we now have multiple outputs in the llike function
def plot_trajectories(mapk_model_name, times, inputs, savedir, type,
    llike_CYTO, llike_PM, llike_CYTO_Rap1KD, llike_PM_Rap1KD, \
    data_CYTO, data_PM, data_CYTO_RAP1i, data_PM_RAP1i,
    data_std_CYTO, data_std_PM, data_std_CYTO_RAP1i, data_std_PM_RAP1i, 
    seed=np.random.default_rng(seed=123)):
    """ Creates prior predictive samples plot of the stimulus response curve.

    Type should be 'prior' or 'posterior'.
    """

    data = {'CYTO':{'llike':llike_CYTO, 'data':data_CYTO, 'data_std':data_std_CYTO},
            'PM':{'llike':llike_PM, 'data':data_PM, 'data_std':data_std_PM},
            'CYTO_Rap1KD':{'llike':llike_CYTO_Rap1KD, 'data':data_CYTO_RAP1i, 'data_std':data_std_CYTO_RAP1i},
            'PM_Rap1KD':{'llike':llike_PM_Rap1KD, 'data':data_PM_RAP1i, 'data_std':data_std_PM_RAP1i}}

    fig_ax = []
    for key in data.keys():
        samples = data[key]['llike']

        if samples.ndim == 3:
            nchains,nsamples,ntime=samples.shape
            samples = np.reshape(samples, (nchains*nsamples, ntime))
        else:
            nsamples,ntime=samples.shape
        
        fig, ax = get_sized_fig_ax(3.0, 1.5)

        # plot the samples
        tr_dict =  {'run':{}, 'timepoint':{}, 'ERK_act':{}}
        names = ['run'+str(i) for i in range(nsamples)]
        idxs = np.linspace(0, (nsamples*times.shape[0]-2), nsamples*times.shape[0]-1)
        cnt = 0
        for i in range(nsamples):
            for j in range(times.shape[0]-1):
                    tr_dict['run'][int(idxs[cnt])] = names[i]
                    tr_dict['timepoint'][int(idxs[cnt])] = times[j+1]
                    tr_dict['ERK_act'][int(idxs[cnt])] = samples[i,j]
                    cnt += 1
        tr_df = pd.DataFrame.from_dict(tr_dict)

        sns.lineplot(data=tr_df,
                    x='timepoint',
                    y='ERK_act',
                    color='c',
                    legend=True,
                    errorbar=('pi', 95), # percentile interval form 2.5th to 97.5th
                    ax=ax)

        # ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0.0, max(times)])

        # plot data
        data_downsample = 3
        ax.errorbar(times[::data_downsample], np.squeeze(data[key]['data'])[::data_downsample], yerr=data[key]['data_std'][::data_downsample], fmt='o', linewidth=1.0, markersize=0.1, color='k')

        ax.set_xlabel('time (min)')
        ax.set_ylabel('ERK activity')

        # save the figure
        fig.savefig(savedir + mapk_model_name + '_' + key + '_' + type + '_predictive.pdf', 
                    bbox_inches='tight', transparent=True)

        # save the samples
        np.save(savedir + mapk_model_name + '_' + key + '_' + type + '_predictive_samples.npy', samples)

        fig_ax.append((fig, ax))
    
    return fig_ax

def build_pymc_model_local(prior_param_dict, diff_params, data, data_sigma, y0, 
                           y0_Rap1KD, output_states, max_time, model_dfrx_ode,
                           simulator=ERK_stim_response):
    """ Builds a pymc model object for the MAPK models.

    Constructs priors for the model, and uses the ERK_stim_response function to 
    generate the stimulus response function and likelihood.
    
    If model is None, the function will use the default model. If a model is s
    pecified, it will use that model_func function to create a PyMC model.

    This is different than the build_pymc_model function in that it creates PyTensor Ops for
    the case with the Rap1 active and inactive which is defined by different initial conditions.
    """

    ####### SOL_OP for the model with Rap1 #######
    # Create jax functions to solve # 
    def sol_op_jax(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted = jax.jit(sol_op_jax)

    
    # Create pytensor Op and register with jax # 
    class StimRespOp(Op):
        def make_node(self, *inputs):
            inputs = [pt.as_tensor_variable(inp) for inp in inputs]
            outputs = [pt.matrix()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result = sol_op_jax_jitted(*inputs)
            if jnp.any(jnp.isnan(jnp.array(result))):
                print('Warning: NaNs in the result. Setting to zeros.')
                result = jnp.zeros_like(result)
            outputs[0][0] = np.asarray(result, dtype="float64")
        
        def grad(self, inputs, output_grads):
            raise NotImplementedError("PyTensor gradient of StimRespOp not implemented")


    # construct Ops and register with jax_funcify
    sol_op = StimRespOp()

    @jax_funcify.register(StimRespOp)
    def sol_op_jax_funcify(op, **kwargs):
        return sol_op_jax
    
    ####### SOL_OP for the model w/0 Rap1 #######
    # Create jax functions to solve # 
    def sol_op_jax_Rap1KD(*params):
        pred, _ = simulator(params, model_dfrx_ode, max_time, y0_Rap1KD, output_states)
        return jnp.vstack((pred))

    # get the jitted versions
    sol_op_jax_jitted_Rap1KD = jax.jit(sol_op_jax_Rap1KD)

    
    # Create pytensor Op and register with jax # 
    class StimRespOp_Rap1KD(Op):
        def make_node(self, *inputs):
            inputs = [pt.as_tensor_variable(inp) for inp in inputs]
            outputs = [pt.matrix()]
            return Apply(self, inputs, outputs)

        def perform(self, node, inputs, outputs):
            result = sol_op_jax_jitted_Rap1KD(*inputs)
            if jnp.any(jnp.isnan(jnp.array(result))):
                print('Warning: NaNs in the result. Setting to zeros.')
                result = jnp.zeros_like(result)
            outputs[0][0] = np.asarray(result, dtype="float64")
        
        def grad(self, inputs, output_grads):
            raise NotImplementedError("PyTensor gradient of StimRespOp not implemented")


    # construct Ops and register with jax_funcify
    sol_op_Rap1KD = StimRespOp_Rap1KD()

    @jax_funcify.register(StimRespOp_Rap1KD)
    def sol_op_Rap1KD_jax_funcify(op, **kwargs):
        return sol_op_jax_Rap1KD
    
    # upack data
    data_CYTO, data_PM, data_CYTO_Rap1KD, data_PM_Rap1KD = data
    data_std_CYTO, data_std_PM, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD = data_sigma

    print(prior_param_dict)
    
    # Construct the PyMC model # 
    model = construct_pm_model(prior_param_dict, diff_params, sol_op, sol_op_Rap1KD,
                               data_CYTO, data_PM, data_std_CYTO, data_std_PM,
                               data_CYTO_Rap1KD, data_PM_Rap1KD, 
                               data_std_CYTO_Rap1KD, data_std_PM_Rap1KD)

    return model

################################################################################
#### PyMC Model Function ####
################################################################################
def construct_pm_model(prior_param_dict, diff_params, sol_op, sol_op_Rap1KD, data_CYTO, data_PM, data_std_CYTO, data_std_PM,
             data_CYTO_Rap1KD, data_PM_Rap1KD, data_std_CYTO_Rap1KD, data_std_PM_Rap1KD):
    """ PyMC model for ERK model with specified parameter different between locs.

        Inputs:
        -   prior_param_dict: dictionary of prior distributions for the parameters
        -   diff_params: list of parameters that differ between compartments
        -   sol_op: PyTensor Op for the model
        -   sol_op_Rap1KD: PyTensor Op for the model with Rap1KD
        -   data_CYTO: data for the cytoplasm compartment
        -   data_PM: data for the plasma membrane compartment
        -   data_std_CYTO: standard deviation of the data for the cytoplasm compartment
        -   data_std_PM: standard deviation of the data for the plasma membrane compartment
        -   data_CYTO_Rap1KD: data for the cytoplasm compartment with Rap1KD
        -   data_PM_Rap1KD: data for the plasma membrane compartment with Rap1KD
        -   data_std_CYTO_Rap1KD: standard deviation of the data for the cytoplasm compartment with Rap1KD
        -   data_std_PM_Rap1KD: standard deviation of the data for the plasma membrane compartment with Rap1KD

        Returns:
        -   model: PyMC model object
    """
    model = pm.Model()
    with model:
        # loop over free params and construct the priors
        priors_CYTO = []
        priors_PM = []
        # create PyMC variables for each parameters in the model
        for key, value in prior_param_dict.items():
            if key in diff_params: # these params will vary between comps, so need two PyMC vars
                info_split = value.split('",')
                prior_CYTO = eval(info_split[0] + '_CYTO",' + info_split[1])
                prior_PM = eval(info_split[0] + '_PM",' + info_split[1])
                priors_CYTO.append(prior_CYTO)
                priors_PM.append(prior_PM)
            else: # these params do not differ between compartments, so just need one PyMC var
                prior = eval(value)
                priors_CYTO.append(prior)
                priors_PM.append(prior)

        # predict full Rap1
        CYTO = pm.Deterministic("CYTO", sol_op(*priors_CYTO))
        PM = pm.Deterministic("PM", sol_op(*priors_PM))
        # Rap1 inhib
        CYTO_Rap1KD = pm.Deterministic("CYTO_Rap1KD", sol_op_Rap1KD(*priors_CYTO))
        PM_Rap1KD = pm.Deterministic("PM_Rap1KD", sol_op_Rap1KD(*priors_PM))
        # normalization factors
        CYTO_norm = pm.Deterministic("CYTO_norm", pt_max(CYTO))
        PM_norm = pm.Deterministic("PM_norm", pt_max(PM))

        # normalized predictions are the actual predictions divided by the max value
        prediction_CYTO = pm.Deterministic("prediction_CYTO", CYTO/CYTO_norm)
        prediction_PM = pm.Deterministic("prediction_PM", PM/PM_norm)
        prediction_CYTO_Rap1KD = pm.Deterministic("prediction_CYTO_Rap1KD", CYTO_Rap1KD/CYTO_norm)
        prediction_PM_Rap1KD = pm.Deterministic("prediction_PM_Rap1KD", PM_Rap1KD/PM_norm) 

        # assume a normal model for the data
        llike_CYTO = pm.Normal("llike_CYTO", mu=prediction_CYTO, sigma=data_std_CYTO, observed=data_CYTO)
        llike_PM = pm.Normal("llike_PM", mu=prediction_PM, sigma=data_std_PM, observed=data_PM)
        llike_CYTO_Rap1KD = pm.Normal("llike_CYTO_Rap1KD", mu=prediction_CYTO_Rap1KD, sigma=data_std_CYTO_Rap1KD, observed=data_CYTO_Rap1KD)
        llike_PM_Rap1KD = pm.Normal("llike_PM_Rap1KD", mu=prediction_PM_Rap1KD, sigma=data_std_PM_Rap1KD, observed=data_PM_Rap1KD)

    return model
##############################
# def arg parsers to take inputs from the command line
##############################
def parse_args(raw_args=None):
    """ function to parse command line arguments
    """
    parser=argparse.ArgumentParser(description="Generate Morris samples for the specified model.")
    parser.add_argument("-model", type=str, help="model to process.")
    parser.add_argument("-free_params", type=str, help="parameters to estimate")
    parser.add_argument("-diff_params", type=str, help="parameters that differ between compartments")
    parser.add_argument("-Rap1_state", type=str, help="Names of inactive Rap1 species.")
    parser.add_argument("-nsamples", type=int, default=1000, help="Number of samples to posterior samples to draw. Defaults to 1000.")
    parser.add_argument("-savedir", type=str, help="Path to save results. Defaults to current directory.")
    parser.add_argument("-input_state", type=str, default='EGF', help="Name of EGF input in the state vector. Defaults to EGF.")
    parser.add_argument("-EGF_conversion_factor", type=float, default=1.0, help="Conversion factor to convert EGF from nM to other units. Defaults to 1.")
    parser.add_argument("-ERK_states", type=str, default=None, help="Names of ERK species to use for inference. Defaults to None.")
    parser.add_argument("-time_conversion_factor", type=float, default=1.0, help="Conversion factor to convert from seconds by division. Default is 1. Mins would be 60")
    parser.add_argument("-prior_family", type=str, default="[['Gamma()',['alpha', 'beta']]]", help="Prior family to use. Defaults to uniform.")
    parser.add_argument("-ncores", type=int, default=1, help="Number of cores to use for multiprocessing. Defaults to None which will use all available cores.")
    parser.add_argument("-nchains", type=int, default=4, help="Number of chains to run. Defaults to 4.")
    parser.add_argument("--skip_sample", action='store_false',default=True)
    parser.add_argument("--skip_prior_sample", action='store_false',default=True)
    parser.add_argument("--negFeedback_KD", action='store_true',default=False)
    parser.add_argument("-negFeedback_state", type=str, default=None)
    parser.add_argument("-negFeedback_param", type=str, default=None)
    parser.add_argument("-rtol", type=float,default=1e-6)
    parser.add_argument("-atol", type=float,default=1e-6)
    parser.add_argument("-upper_prior_mult", type=float,default=1e3)
    parser.add_argument("-lower_prior_mult", type=float,default=1e-3)

    args=parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    """ main function to execute command line script functionality.
    """
    seed = np.random.default_rng(seed=123)
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
    # load in the training data
    data_file = '../../../results/MAPK/Keyes_et_al_2020-fig1-data1-v2-'

    inputs_CYTO, data_CYTO, data_std_CYTO, times_CYTO \
        = load_data_json(data_file+'CYTO_CYTOmax.json', data_std=True, time=True)
    _, data_PM, data_std_PM, times_PM \
        = load_data_json(data_file+'PM_PMmax.json', data_std=True, time=True)

    data_file = '../../../results/MAPK/Keyes_et_al_2020-fig3-data1-v2-'
    _, data_CYTO_RAP1i, data_std_CYTO_RAP1i, \
        times_CYTO_RAP1i = load_data_json(data_file+'CYTO_RAP1inhib_CYTOmax.json', \
        data_std=True, time=True)
    _, data_PM_RAP1i, data_std_PM_RAP1i, times_PM_RAP1i \
        = load_data_json(data_file+'PM_RAP1inhib_PMmax.json', \
        data_std=True, time=True)

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
    data_CYTO = data_CYTO_interp.evaluate(times)
    data_std_CYTO = data_std_CYTO_interp.evaluate(times)
    data_PM = data_PM_interp.evaluate(times)
    data_std_PM = data_std_PM_interp.evaluate(times)

    data_CYTO_RAP1i = data_CYTO_RAP1i_interp.evaluate(times)
    data_std_CYTO_RAP1i = data_std_CYTO_RAP1i_interp.evaluate(times)
    data_PM_RAP1i = data_PM_RAP1i_interp.evaluate(times)
    data_std_PM_RAP1i = data_std_PM_RAP1i_interp.evaluate(times)


    # convert EGF to required units
    # inputs are the same in both compartments, so just use CYTO
    inputs_native_units = inputs_CYTO * args.EGF_conversion_factor

    # get the params to sample
    analyze_params = args.free_params.split(',')
    free_param_idxs = [list(p_dict.keys()).index(p) for p in analyze_params]

    diff_params = args.diff_params.split(',')
    # for some reason \r is getting added, so remove it
    diff_params = [p.strip('\r') for p in diff_params]

    # get the EGF index and ERK indices
    state_names = list(y0_dict.keys())
    EGF_idx = state_names.index(args.input_state)
    ERK_indices = [state_names.index(s) for s in args.ERK_states.split(',')]

    # construct the strings to make priors and constants
    prior_param_dict = set_prior_params(args.model, list(p_dict.keys()), plist, free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family, savedir=args.savedir, saveplot=False)
    
    # make simulator lambda function that solves at correct times with the time conversion factor taken into account]
    # NOTE: use times[1:] to avoind issues associated with included t=0 point
    @jax.jit
    def ERK_stim_traj(p, model, max_time, y0, output_states):
        traj = solve_traj(model, y0, p, max_time, ERK_indices, times[1:]/args.time_conversion_factor, args.rtol, args.atol)

        return [traj], traj

    # make initial conditions that reflect the inputs
    y0_EGF_ins = construct_y0_EGF_inputs(inputs_native_units, np.array([y0]), EGF_idx)

    # constuct initial condition with Rap1 kockdown
    y0_Rap1_knockdown = y0_dict.copy()
    y0_Rap1_knockdown[args.Rap1_state] = 0.0
    y0_Rap1KD = tuple(y0_Rap1_knockdown.values())
    y0_EGF_ins_Rap1_KD = construct_y0_EGF_inputs(inputs_native_units, np.array([y0_Rap1KD]), EGF_idx)

    # # construct the pymc model
    # # Note: We do not use the build_pymc_model function, because we need to 
    # #   build a model that runs the simulator three times for each input level
    # # NOTE: use data_CYTO[1:], etc to avoid issues associated with included t=0 point
    # try:
    #     model_func = lambda prior_param_dict, sol_op, sol_op_Rap1KD, data, \
    #         data_std: eval(args.pymc_model)(prior_param_dict, sol_op, \
    #         sol_op_Rap1KD, [data_CYTO[1:]], [data_PM[1:]], [data_std_CYTO[1:]], \
    #             [data_std_PM[1:]],  [data_CYTO_RAP1i[1:]], [data_PM_RAP1i[1:]], \
    #             [data_std_CYTO_RAP1i[1:]], [data_std_PM_RAP1i[1:]])
    # except OSError as e:
    #     print('Warning Pymc model {} not found'.format(args.pymc_model))
    #     raise

    data = ([data_CYTO[1:]], [data_PM[1:]], [data_CYTO_RAP1i[1:]], [data_PM_RAP1i[1:]])
    data_sigma = ([data_std_CYTO[1:]], [data_std_PM[1:]], [data_std_CYTO_RAP1i[1:]], [data_std_PM_RAP1i[1:]])
        
    pymc_model = build_pymc_model_local(prior_param_dict, diff_params, data, 
                                        data_sigma, y0_EGF_ins[0], 
                                        y0_EGF_ins_Rap1_KD[0], ERK_indices, 
                                        np.max(times/args.time_conversion_factor), 
                                        diffrax.ODETerm(model), 
                                        simulator=ERK_stim_traj)
    

    
    if args.skip_prior_sample:
        # sample from the posterior predictive
        with pymc_model:
            prior_predictive = pm.sample_prior_predictive(samples=500, random_seed=seed)
        
        prior_predictive.to_json(args.savedir + args.model + '_prior_samples.json')

        # extract llike values
        prior_llike_CYTO = np.squeeze(prior_predictive.prior_predictive['llike_CYTO'].values)
        prior_llike_PM = np.squeeze(prior_predictive.prior_predictive['llike_PM'].values)
        prior_llike_CYTO_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_CYTO_Rap1KD'].values)
        prior_llike_PM_Rap1KD = np.squeeze(prior_predictive.prior_predictive['llike_PM_Rap1KD'].values)

        plot_trajectories(args.model, times/60, inputs_CYTO, args.savedir, 'prior',
        prior_llike_CYTO, prior_llike_PM, prior_llike_CYTO_Rap1KD, prior_llike_PM_Rap1KD, data_CYTO, 
        data_PM, data_CYTO_RAP1i, data_PM_RAP1i, data_std_CYTO, data_std_PM, 
        data_std_CYTO_RAP1i, data_std_PM_RAP1i)

    # SMC sampling
    if args.skip_sample:
        posterior_idata = smc_pymc(pymc_model, args.model, args.savedir, 
                    nsamples=args.nsamples, ncores=args.ncores, chains=args.nchains,seed=seed)
    else:
        posterior_idata, _ = load_smc_samples_to_idata(args.savedir + args.model + '_smc_samples.json')
    
    # trace plots and diagnostics
    plot_sampling_trace_diagnoses(posterior_idata, args.savedir, args.model)

    # posterior predictive samples
    with pymc_model:
        # sample from the posterior predictive
        posterior_predictive = pm.sample_posterior_predictive(posterior_idata,random_seed=seed)
        
    posterior_predictive.to_json(args.savedir + args.model + '_posterior_samples.json')
    
    # extract llike values
    llike_CYTO = np.squeeze(posterior_predictive.posterior_predictive['llike_CYTO'].values)
    llike_PM = np.squeeze(posterior_predictive.posterior_predictive['llike_PM'].values)
    llike_CYTO_Rap1KD = np.squeeze(posterior_predictive.posterior_predictive['llike_CYTO_Rap1KD'].values)
    llike_PM_Rap1KD = np.squeeze(posterior_predictive.posterior_predictive['llike_PM_Rap1KD'].values)

    # make the plots
    plot_trajectories(args.model, times/60, inputs_CYTO, args.savedir, 'posterior',
        llike_CYTO, llike_PM, llike_CYTO_Rap1KD, llike_PM_Rap1KD, data_CYTO, 
        data_PM, data_CYTO_RAP1i, data_PM_RAP1i, data_std_CYTO, data_std_PM, 
        data_std_CYTO_RAP1i, data_std_PM_RAP1i)
    
    # Run posterior predictive simulaitons w/o neg feed back and w/o neg feedback and Rap1KD
    if args.negFeedback_KD and ((args.negFeedback_state is not None) or (args.negFeedback_param is not None)):
        print('Running posterior predictive simulations w/o negative feedback...')

        if args.negFeedback_state is not None: # knockdown the negative feedback by setting an initial condition to 0
            # constuct initial condition with nedFeedBack kockdown % Rap1 KD
            y0_negFeedback_knockdown = y0_dict.copy()
            y0_negFeedback_knockdown[args.negFeedback_state] = 0.0
            y0_negFeedbackKD = tuple(y0_negFeedback_knockdown.values())
            y0_EGF_ins_negFeedback_KD = construct_y0_EGF_inputs(inputs_native_units, np.array([y0_negFeedbackKD]), EGF_idx)

            y0_negFeedback_Rap1_knockdown = y0_dict.copy()
            y0_negFeedback_knockdown[args.negFeedback_state] = 0.0
            y0_negFeedback_Rap1_knockdown[args.Rap1_state] = 0.0
            y0_negFeedback_Rap1KD = tuple(y0_negFeedback_Rap1_knockdown.values())
            y0_EGF_ins_negFeedback_Rap1_KD = construct_y0_EGF_inputs(inputs_native_units, np.array([y0_negFeedback_Rap1KD]), EGF_idx)
        elif args.negFeedback_param is not None: # knockdown the negative feedback by setting a constant param 0
            # first copy ics (these are the same as not knocking down the negative feedback)
            y0_EGF_ins_negFeedback_KD = y0_EGF_ins.copy()
            y0_EGF_ins_negFeedback_Rap1_KD = y0_EGF_ins_Rap1_KD.copy()

            # now construct a new prior dict with the negative feedback param set to 0
            # p_dict_negFeedback = p_dict.copy()
            # p_dict_negFeedback[args.negFeedback_param] = 0.0 # set the desired parameter to zero
            # plist_negFeedback = [p_dict_negFeedback[p] for p in list(p_dict_negFeedback.keys())]
            # prior_param_dict = set_prior_params(args.model, list(p_dict_negFeedback.keys()), plist_negFeedback, free_param_idxs, upper_mult=args.upper_prior_mult, lower_mult=args.lower_prior_mult, prior_family=args.prior_family, savedir=args.savedir, saveplot=False)
            prior_param_dict[args.negFeedback_param] = 'pm.ConstantData("' + args.negFeedback_param + '", 0.0)'
             
        # construct the pymc model
        # here we slightly abuse the notation within the model function to pass the negative feedback state
        # the llike for the negFeedback knockdown is 'llike_CYTO' and 'llike_PM' whereas
        # that for the negFeedback and Rap1KD is 'llike_CYTO_Rap1KD' and 'llike_PM_Rap1KD'
        pymc_model_negFeedbackKD = build_pymc_model_local(prior_param_dict, diff_params, data, [1e-14,1e-14,1e-14,1e-14], 
                                    y0_EGF_ins_negFeedback_KD[0], y0_EGF_ins_negFeedback_Rap1_KD[0], 
                                    ERK_indices, np.max(times/args.time_conversion_factor), 
                                    diffrax.ODETerm(model), simulator=ERK_stim_traj)
    
        # create new idata without log_loglikelihood 
        posterior_idata_negFeedbackKD = posterior_idata.copy()
        del posterior_idata_negFeedbackKD.log_likelihood
        del posterior_idata_negFeedbackKD.posterior['prediction_CYTO']
        del posterior_idata_negFeedbackKD.posterior['prediction_PM']
        del posterior_idata_negFeedbackKD.posterior['prediction_CYTO_Rap1KD']
        del posterior_idata_negFeedbackKD.posterior['prediction_PM_Rap1KD']
        del posterior_idata_negFeedbackKD.posterior['CYTO_norm']
        del posterior_idata_negFeedbackKD.posterior['PM_norm']

        # run posterior predictive simulations
        with pymc_model_negFeedbackKD:
            # sample from the posterior predictive
            posterior_predictive_negFeedbackKD = pm.sample_posterior_predictive(posterior_idata,random_seed=seed, predictions=True, var_names=['llike_CYTO', 'llike_PM', 'llike_CYTO_Rap1KD', 'llike_PM_Rap1KD', 'prediction_CYTO', 'prediction_PM', 'prediction_CYTO_Rap1KD', 'prediction_PM_Rap1KD', 'CYTO_norm', 'PM_norm'])
        
        posterior_predictive_negFeedbackKD.to_json(args.savedir + args.model + '_posterior_nedFeedBackKD_samples.json')

        # extract llike values
        llike_CYTO_negFeedbackKD = np.squeeze(posterior_predictive_negFeedbackKD.predictions['llike_CYTO'].values)
        llike_PM_negFeedbackKD = np.squeeze(posterior_predictive_negFeedbackKD.predictions['llike_PM'].values)
        llike_CYTO_Rap1KD_negFeedbackKD = np.squeeze(posterior_predictive_negFeedbackKD.predictions['llike_CYTO_Rap1KD'].values)
        llike_PM_Rap1KD_negFeedbackKD = np.squeeze(posterior_predictive_negFeedbackKD.predictions['llike_PM_Rap1KD'].values)

        # make the plots we dont want to show the data so we pass nan
        plot_trajectories(args.model, times/60, inputs_CYTO, args.savedir, 'posterior_negFeedbackKD',
            llike_CYTO_negFeedbackKD, llike_PM_negFeedbackKD, 
            llike_CYTO_Rap1KD_negFeedbackKD, llike_PM_Rap1KD_negFeedbackKD, 
            np.nan*np.ones_like(data_CYTO), np.nan*np.ones_like(data_PM), 
            np.nan*np.ones_like(data_CYTO_RAP1i), np.nan*np.ones_like(data_PM_RAP1i), 
            np.nan*np.ones_like(data_std_CYTO), np.nan*np.ones_like(data_std_PM), 
            np.nan*np.ones_like(data_std_CYTO_RAP1i), np.nan*np.ones_like(data_std_PM_RAP1i))

    elif args.negFeedback_KD and args.negFeedback_state==None:
        print('Warning: Please provide the name of the negative feedback state')
    
    print('Completed {}'.format(args.model))

if __name__ == '__main__':
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn')
    main()
