#Study_3 - SimulationStudy 

library(cmdstanr)
library(parallel) # for detectCores()
library(ff) # for moving directories
library(bayesplot)
library(optparse) # for input arguments

## IMPORTANT: Set cmdstan location:
# Create a local setup.local.R file that calls set_cmdstan_path():
# $ cp setup.template.R setup.local.R
# Set the cmdstan location in setup.local.R. This preserves the location
# even after future code changes
source("setup.local.R")

argument_list = list (
  make_option(c('-w', '--warmup'), type='integer', default=100,
              help='Number warmup iterations', metavar='integer'),
  make_option(c('-s', '--sampling'), type='integer', default=400,
              help='Number sampling iterations', metavar='integer'),
  make_option(c('-t', '--threads_per_chain'), type='integer', default=detectCores(),
              help='threads_per_chain', metavar='integer'),
  make_option(c('-l', '--subject_range_lower'), type='integer', default=1,
              help='where to start sampling', metavar='integer'),
  make_option(c('-u', '--subject_range_upper'), type='integer', default=2,
              help='where to end sampling', metavar='integer'),
  make_option(c('-m', '--model'), type='character', default='full',
              help='model: basic, basic_fix, basic_cens', metavar='character')
)
arg_parser = OptionParser(option_list=argument_list)
args = parse_args(arg_parser)

## Set configuration and choose which setup to sample with input arguments
max_treedepth = 5 # 5 or 10
subject_range = args$subject_range_lower:args$subject_range_upper   #set subject range for each analysis-group

model = args$model
if (args$model %in% c('basic', 'basic_cens')) { #set data_dir according to S3_simulate.R
  data_dir = paste0('S3_truncated_', model)
} else if (args$model %in% c('basic_fix', 'basic_fix_new')) { #new: in model with *CDF(inf)
  data_dir = paste0('S3_truncated_', args$model)
  model = 'basic'
} else if (args$model %in% c('basic_fix_cdf1_cdf0', 'basic_fix_cdf1_cdf0_log_sum_exp')) { #new: in model with *CDF(inf)
  data_dir = paste0('S3_truncated_', args$model)
} else {
  data_dir = 'S3_truncated'
}

NumberTrials = c(100, 500)

print(paste("Fitting subjects:", str(subject_range))) 
for (subj in subject_range) {
  for (NTrials in 1:length(NumberTrials)){
    subj_dir = file.path(data_dir, formatC(subj, width = 4, flag = "0"), NumberTrials[NTrials])
    output_dir = file.path(subj_dir, paste0('fit_', args$model))
    if ("inference.txt" %in% list.files(output_dir)) { # check if fit was already done
      next
    }
    print(paste0("Subject ", subj, ", trials: ", NumberTrials[NTrials]))
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }
    sim_data = read.csv(file.path(subj_dir, "data.csv"))
    min_rt = min(sim_data$rt)
    params = read.csv(file.path(data_dir, formatC(subj, width = 4, flag = '0'), 'params.csv'))
    colnames(sim_data) = c('trial', "cnd", "resp", "rt")
    censored = as.numeric(sim_data$rt>params$upper_bound[NTrials])
    N_cens_resp_0_cnd_1 = sum(sim_data[sim_data$rt>params$upper_bound[NTrials],]$resp == 0 & sim_data[sim_data$rt>params$upper_bound[NTrials],]$cnd == 1)
    N_cens_resp_0_cnd_2 = sum(sim_data[sim_data$rt>params$upper_bound[NTrials],]$resp == 0 & sim_data[sim_data$rt>params$upper_bound[NTrials],]$cnd == 2)
    N_cens_resp_1_cnd_1 = sum(sim_data[sim_data$rt>params$upper_bound[NTrials],]$resp == 1 & sim_data[sim_data$rt>params$upper_bound[NTrials],]$cnd == 1)
    N_cens_resp_1_cnd_2 = sum(sim_data[sim_data$rt>params$upper_bound[NTrials],]$resp == 1 & sim_data[sim_data$rt>params$upper_bound[NTrials],]$cnd == 2)
    if (args$model == 'basic_cens' & sum(censored) != 0) {
      sim_data[sim_data$rt>params$upper_bound[NTrials],]$rt = params$upper_bound[NTrials] # censoring
    }
    
    stan_data = list(
      N = nrow(sim_data),     # Number of Trials
      cnd = sim_data$cnd,     # stimulus type (1,2)
      rt = sim_data$rt,       # rt in seconds!
      resp = sim_data$resp,   # response (0=lower threshold, 1=upper threshold)
      upper_bound = params$upper_bound[NTrials],
      censored = censored,        # only relevant for censored model
      N_cens_resp_0_cnd_1 = N_cens_resp_0_cnd_1,  # only relevant for censored model, number of response = 0 and condition = 1 in censored data
      N_cens_resp_0_cnd_2 = N_cens_resp_0_cnd_2,  # only relevant for censored model, number of response = 0 and condition = 2 in censored data
      N_cens_resp_1_cnd_1 = N_cens_resp_1_cnd_1,  # only relevant for censored model, number of response = 1 and condition = 1 in censored data
      N_cens_resp_1_cnd_2 = N_cens_resp_1_cnd_2,   # only relevant for censored model, number of response = 1 and condition = 2 in censored data
      min_rt = min_rt
    )
    
    # configure init values and fixed values for different models
    init.stan = function(){
      L = list()
      for (i in 1:4) {
        L[[i]] = list(
          # set initial values to plausible values
          a = 1 + runif(1, -0.2, 0.2),
          v_m_pos = 2.0 + runif(2, -0.2, 0.2),
          zr_m = 0.5 + runif(1, -0.1, 0.1),
          t0_m = runif(1, 0.2 + 1e-9, min(c(min_rt-1e-5, 0.3))),
          
          v_s = 1 + runif(1, -0.2, 0.2),
          zr_s = 0.1 + runif(1, -0.05, 0.05),
          t0_s = 0.2 + runif(1, -0.05, 0.05)
        )
      }
      return (L)
    }


    # compile the model
    file = paste0("S3_", model, ".stan")
    mc = cmdstan_model(file, cpp_options = list(stan_threads = T))

    # do the sampling
    L = init.stan()
    m = mc$sample(data=stan_data,
                init = L,
                max_treedepth = max_treedepth,
                adapt_delta = 0.8,
                refresh = 50,
                iter_sampling = args$sampling,
                iter_warmup = args$warmup,
                chains = 4,
                parallel_chains = 4,
                threads_per_chain = args$threads_per_chain,
                output_dir = output_dir,
                output_basename = 'fit',
                save_warmup = TRUE)

    summary = m$summary()
    summary = summary[summary$variable != "lp__",]
    sink(file.path(output_dir, "inference.txt"))
    print(m$cmdstan_summary())
    sink()
    
    # check whether diagnotic criteria for NEff and r_hat are fine
    NEff_val = min(summary$ess_bulk)
    Rhat_val = max(summary$rhat)
    
    # if not, move the  directory
    if (NEff_val < 400 | Rhat_val > 1.01) {
      failed_dir = paste0("failed_", output_dir, format(Sys.time(), format="_%y%m%d-%H%M%S"))
      dir.create(failed_dir, recursive = TRUE, showWarnings = FALSE)
      file.move(output_dir, failed_dir)
      print(paste0('Moved to failed_fits.'))
      print(paste0('NEff = ', NEff_val, ', Rhat_val = ', Rhat_val))
    } #end if move
  } #end for NTrials
} #end subj
